"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

import h5py
import robosuite.utils.transform_utils as T

import wandb
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor, get_ecot_action, get_vla_action, get_tcot_action, get_fine_action, get_fine_action_withscore
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

import torch
from peft import LoraConfig, PeftModel
from transformers import AutoModelForVision2Seq
import json
import cv2
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import  TcotSearchForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
# python experiments/robot/libero/run_libero_eval_ecot.py   --model_family openvla   --pretrained_checkpoint exp_model/openvla-7b+libero_spatial_ecot_s5000/   --task_suite_name libero_spatial --center_crop True
# python experiments/robot/libero/run_libero_eval_ecot.py   --model_family openvla   --pretrained_checkpoint openvla-7b/   --task_suite_name libero_spatial --center_crop True
@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = "/home/cvailab/codebase/embodied-CoT/openvla/openvla-7b"     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_10"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 20                    # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = 'libero_10_vla_generate_failcase_200try_s20000'                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    max_tcot_step: int = 0

    #################
    # generate failure case
    #################
    libero_target_dir: str = '/home/cvailab/codebase/datasets/LIBERO/libero_10_failure_200try'
    record_failure_case: bool = True

    # fmt: on

DEVICE = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")

def draw_traj(img, generated_txt, horizon):
    str_traj = generated_txt.split("Trajectory: ")[-1].split("Action: ")[0]
    if str_traj[0] != '[':
        str_traj = '['+str_traj
    traj = json.loads(str_traj)
    traj = traj[:2*horizon]
    x = np.array(traj[::2]) * 224 // 256
    y = np.array(traj[1::2]) * 224 // 256
    num_point = min(len(x), len(y))
    for i in range(num_point):
        cv2.circle(img, (int(x[i]), int(y[i])), radius=2, color=(0, 0, 255), thickness=-1)
    return img

def draw_text(img, text):
    # 定义文字内容、位置、字体、大小、颜色和粗细
    position = (50, 50)  # 文字在图片上的位置（左上角）
    font = cv2.FONT_HERSHEY_SIMPLEX  # 字体类型
    font_scale = 1  # 字体大小
    text_color = (255, 255, 255)  # 文字颜色（BGR格式，白色）
    thickness = 2  # 字体粗细

    # 在图片上添加文字
    cv2.putText(img, text, position, font, font_scale, text_color, thickness)
    return img

def draw_line(img, score_list):
    points = np.array([[int(x*1.2), int(50 - y/6*50)] for x,y in enumerate(score_list)])
    img = cv2.polylines(img, [points], isClosed=False, color=(0, 0, 255), thickness=2)
    return img

@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    vla_path = cfg.pretrained_checkpoint
    # Load model
    #model = get_model(cfg)

    # Load model from lora
    using_lora = True
    if using_lora:
        adapter_dir = "/home/cvailab/codebase/embodied-CoT/openvla/exp_model/lora_model/openvla-7b+liber_o10+openvla+baseline_s10000+cofintune0.0/step_20000"
        print(f'loading lora model from {adapter_dir} for inference, base model from {vla_path}')
        
    else:
        print(f'loading base model from {vla_path}')
        

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    #AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    AutoModelForVision2Seq.register(OpenVLAConfig, TcotSearchForActionPrediction)

    base_vla = AutoModelForVision2Seq.from_pretrained(
        vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    )
    if using_lora:
        model = PeftModel.from_pretrained(base_vla, adapter_dir)
    else:
        model = base_vla #if not using lora adaptor
    
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        model = model.to(DEVICE)
        

    # Load dataset stats used during finetuning (for action un-normalization).
    dataset_statistics_path = os.path.join(cfg.pretrained_checkpoint, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        model.norm_stats = norm_stats
    #print('model.norm_stats: ', model.norm_stats)


    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize local logging
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")
    if using_lora:
        log_file.write(f'loading lora model from {adapter_dir} for inference, base model from {vla_path}\n')
    else:
        log_file.write(f'loading base model from {vla_path}\n')
    log_file.write(f'model.max_freezing_time: {model.max_freezing_time}\n')

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Setup
    if cfg.record_failure_case:
        num_replays = 0
        num_success = 0
        num_failure = 0
        num_noops = 0
        # Prepare JSON file to record success/false and initial states per episode
        metainfo_json_dict = {}
        metainfo_json_out_path = f"./experiments/robot/libero/{cfg.task_suite_name}_metainfo.json"
        with open(metainfo_json_out_path, "w") as f:
            # Just test that we can write to this file (we overwrite it later)
            json.dump(metainfo_json_dict, f)

    # Start evaluation
    total_episodes, total_successes, total_retry_successes = 0, 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        # Create new HDF5 file for regenerated demos
        if cfg.record_failure_case:
            new_data_path = os.path.join(cfg.libero_target_dir, f"{task.name}_demo.hdf5")
            new_data_file = h5py.File(new_data_path, "w")
            grp = new_data_file.create_group("data")

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):

            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 180 #220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 400 #520  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps

            # Set up new data lists
            states = []
            actions = []
            ee_states = []
            gripper_states = []
            joint_states = []
            robot_states = []
            agentview_images = []
            eye_in_hand_images = []

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            model.max_freezing_time = max_steps
            model.time_frozen = 0
            print(f'model.max_freezing_time: {model.max_freezing_time}')

            strategy = 'vla' # [vla, tcot]
            print('strategy:', strategy)
            
            tcot_score_list, vla_score_list = [], []
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    img = get_libero_image(obs, resize_size)

                    # Prepare observations dict
                    # Note: OpenVLA does not take proprio state as input
                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }

                    if  strategy=='tcot':
                        action, generated_text, action_score = get_tcot_action(
                            cfg,
                            model,
                            observation,
                            task_description,
                            processor=processor,
                            center_crop=True,
                            output_score=True
                        )
                        tcot_score_list.append(action_score.cpu().numpy())
                        #print(f"t={t}: tcot_action_score={action_score}")
                        img = draw_traj(img, generated_text, horizon=30)
                        img = draw_line(img, tcot_score_list)
                    else:
                        action, action_score = get_fine_action_withscore(
                            cfg,
                            model,
                            observation,
                            task_description,
                            processor=processor,
                            center_crop=True,
                        )
                        #img = draw_text(img, str(action_score.cpu().numpy()))
                        vla_score_list.append(action_score.cpu().numpy())
                        #img = draw_line(img, vla_score_list)
                        #print(f"t={t}: vla_action_score={action_score}")


                    #img = draw_traj(img, generated_text, horizon=30)
                    replay_images.append(img)

                    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                    action = normalize_gripper_action(action, binarize=True)

                    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)

                    
                    if cfg.record_failure_case:
                        # For all other timesteps, get state from environment and record it
                        states.append(env.sim.get_state().flatten())
                        robot_states.append(
                            np.concatenate([obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]])
                        )
                        # Record original action (from demo)
                        actions.append(action)

                        # Record data returned by environment
                        if "robot0_gripper_qpos" in obs:
                            gripper_states.append(obs["robot0_gripper_qpos"])
                        joint_states.append(obs["robot0_joint_pos"])
                        ee_states.append(
                            np.hstack(
                                (
                                    obs["robot0_eef_pos"],
                                    T.quat2axisangle(obs["robot0_eef_quat"]),
                                )
                            )
                        )
                        agentview_images.append(obs["agentview_image"])
                        eye_in_hand_images.append(obs["robot0_eye_in_hand_image"])

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                    if t == (max_steps + cfg.num_steps_wait) and strategy=='tcot':
                        strategy = 'tcot'
                        print(f'openvla fail at step {t} and retry with {strategy} .................................')
                        log_file.write(f'openvla fail at step {t} and retry with {strategy} .................................\n')
                        # start retry  Reset environment
                        env.reset()
                        obs = env.set_init_state(initial_states[episode_idx])
                        t = 0
                        #max_tcot_step = max_steps + cfg.num_steps_wait #change from 0->230


                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            task_episodes += 1
            total_episodes += 1
            if strategy=='tcot' and done:
                total_retry_successes +=1

            if cfg.record_failure_case and not done:
                dones = np.zeros(len(actions)).astype(np.uint8)
                rewards = np.zeros(len(actions)).astype(np.uint8)
                num_failure += 1
                assert len(actions) == len(agentview_images)

                ep_data_grp = grp.create_group(f"demo_{episode_idx}")
                obs_grp = ep_data_grp.create_group("obs")
                obs_grp.create_dataset("gripper_states", data=np.stack(gripper_states, axis=0))
                obs_grp.create_dataset("joint_states", data=np.stack(joint_states, axis=0))
                obs_grp.create_dataset("ee_states", data=np.stack(ee_states, axis=0))
                obs_grp.create_dataset("ee_pos", data=np.stack(ee_states, axis=0)[:, :3])
                obs_grp.create_dataset("ee_ori", data=np.stack(ee_states, axis=0)[:, 3:])
                obs_grp.create_dataset("agentview_rgb", data=np.stack(agentview_images, axis=0))
                obs_grp.create_dataset("eye_in_hand_rgb", data=np.stack(eye_in_hand_images, axis=0))
                ep_data_grp.create_dataset("actions", data=actions)
                ep_data_grp.create_dataset("states", data=np.stack(states))
                ep_data_grp.create_dataset("robot_states", data=np.stack(robot_states, axis=0))
                ep_data_grp.create_dataset("rewards", data=rewards)
                ep_data_grp.create_dataset("dones", data=dones)
            
                task_key = task_description.replace(" ", "_")
                episode_key = f"demo_{episode_idx}"
                if task_key not in metainfo_json_dict:
                    metainfo_json_dict[task_key] = {}
                if episode_key not in metainfo_json_dict[task_key]:
                    metainfo_json_dict[task_key][episode_key] = {}
                metainfo_json_dict[task_key][episode_key]["success"] = bool(done)
                metainfo_json_dict[task_key][episode_key]["initial_state"] = initial_states[episode_idx].tolist()
                # Write metainfo dict to JSON file
                # (We repeatedly overwrite, rather than doing this once at the end, just in case the script crashes midway)
                with open(metainfo_json_out_path, "w") as f:
                    json.dump(metainfo_json_dict, f, indent=2)
                # Count total number of successful replays so far
                print(
                    f"Recording Failure Case: Total # episodes replayed: {total_episodes}, Total # successes: {num_success}, Total # failure: {num_failure})"
                )


            # Save a replay video of the episode
            save_rollout_video(
                replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file, note=cfg.run_id_note
            )

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            print(f"total_retry_successes: {total_retry_successes}")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.write(f"total_retry_successes: {total_retry_successes}")
            log_file.flush()


        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )
        if cfg.record_failure_case:
            # Close HDF5 files
            new_data_file.close()
            print(f"Saved regenerated demos for task '{task_description}' at: {new_data_path}")

    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    if cfg.record_failure_case:
        print(f"Dataset regeneration complete! Saved new failure dataset at: {cfg.libero_target_dir}")
        print(f"Saved metainfo JSON at: {metainfo_json_out_path}")

if __name__ == "__main__":

    eval_libero()
