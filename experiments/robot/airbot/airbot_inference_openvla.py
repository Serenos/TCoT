import requests
import json_numpy
json_numpy.patch()
import json
import sys
import argparse
import numpy as np
import time
from airbot_utils.airbot_play_real_env import make_env, get_image, move_arms, move_grippers
from typing import List
import airbot
from airbot_utils.custom_robot import AssembledRobot
from tqdm import tqdm
import torch
import cv2
from scipy.spatial.transform import Rotation

from PIL import Image
import tensorflow as tf
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import os
from experiments.robot.robot_utils import (
    DATE_TIME,
    set_seed_everywhere,
)

from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import  TcotSearchForActionPrediction, OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from peft import LoraConfig, PeftModel
from transformers import AutoModelForVision2Seq

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
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = 'airbot_mix_pickcone2'                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    max_tcot_step: int = 0

    command: str = "Lift the plate"
    unnorm_key: str = "airbot_mix"
    camera_web: str = "0"
    camera_mv: str = ""
    can: str = "can1"
    eef_mode: str = "gripper"
    time_steps: int = 50
    # fmt: on

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def get_action(image=np.zeros((224,224,3), dtype=np.uint8), command='do something', unnorm_key='bridge_orig', session=None):
    # unnorm_key should be changed accordingly
    action = session.post(
        "http://localhost:8000/act",
        json={"image": image, "instruction": command, "unnorm_key" : unnorm_key}
    ).json()
    return action

def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image

def get_processor(cfg):
    """Get VLA model's Hugging Face processor."""
    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)
    return processor


def get_vla_action(vla, cfg, image, processor, task_label, center_crop=True):
    """Generates an action with the VLA policy."""
    #image = image[:,:,::-1]

    image = Image.fromarray(image)
    image = image.convert("RGB")

    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
    #            the original height and width by sqrt(0.9) -- not 0.9!
    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        # Convert to TF Tensor and record original data type (should be tf.uint8)
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype

        # Convert to data type tf.float32 and values between [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Crop and then resize back to original size
        image = crop_and_resize(image, crop_scale, batch_size)

        # Convert back to original data type
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

        # Convert back to PIL Image
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")


    prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
    print(prompt)

    # Process inputs.
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)


    # Run OpenVLA Inference
    start_time = time.time()

    #torch.manual_seed(0)
    action, generated_ids = vla.predict_action(**inputs, unnorm_key=cfg.unnorm_key, do_sample=False)
    #print(f"Time: {time.time() - start_time:.4f} || Action: {action}")

    # Get action.
    # action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    return action

def resize_image(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    assert isinstance(resize_size, tuple)
    # Resize to image size expected by model
    img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img

# Function to convert roll, pitch, yaw to quaternion
def euler_to_quaternion(roll, pitch, yaw):
    r = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=False)
    return r.as_quat()

if __name__ == "__main__":
    cfg = GenerateConfig()

    camera_names = ['1']
    camera_web = cfg.camera_web
    camera_mv = cfg.camera_mv
    if camera_mv == "":
        camera_indices = camera_web
    elif camera_web == "":
        camera_indices = camera_mv
    else: camera_indices = camera_mv + ',' + camera_web
    camera_mv_num = len(camera_mv.split(",")) if camera_mv!="" else 0
    camera_web_num = len(camera_web.split(",")) if camera_web!="" else 0
    camera_mask = ["mv"] * camera_mv_num + ["web"] * camera_web_num
    cameras = {name: int(index) for name, index in zip(camera_names, camera_indices.split(','))}
    cameras["mask"] = camera_mask
    
    # init VLA model
    set_seed_everywhere(cfg.seed)
    # Load model from lora
    using_lora = True
    if using_lora:
        adapter_dir = "/home/cvailab/codebase/embodied-CoT/openvla/exp_model/lora_model_airbot/openvla-7b+airbot_mix+openvla+pick_plate2_skip1stop1+cofintune0.0/step_500"
        print(f'loading lora model from {adapter_dir} for inference, base model from {cfg.pretrained_checkpoint}')
        
    else:
        print(f'loading base model from {cfg.pretrained_checkpoint}')
        
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    base_vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True, attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
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
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize local logging
    run_id = f"EVAL-{cfg.unnorm_key}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")
    if using_lora:
        log_file.write(f'loading lora model from {adapter_dir} for inference, base model from {cfg.pretrained_checkpoint}\n')
    else:
        log_file.write(f'loading base model from {cfg.pretrained_checkpoint}\n')

    # init robots
    robot_instances:List[AssembledRobot] = []

    # modify the path to the airbot_play urdf file
    vel = 2.0
    fps = 30
    joint_num = 7
    robot_num = 1
    start_joint = [0] * 7
    can_list = cfg.can.split(',')
    eef_mode = cfg.eef_mode

    # UPSAMPLE = 100
    # control_freq = 10
    # DT = 1.0 / control_freq
    
    image_mode = 0
    for index, can in enumerate(can_list):
        #airbot_player = airbot.create_agent("down", can, vel, eef_mode) # urdf_path
        airbot_player = airbot.create_agent(direction="down", can_interface="can1", end_mode="gripper") 
        # airbot_player.set_target_pose([0.3,0,0.2], [0,0,0,1])
        # airbot_player.set_target_end(0) airbot_player.get_current_pose()
        # time.sleep(5)
        robot_instances.append(AssembledRobot(airbot_player, 1/fps, 
                                                start_joint[joint_num*index:joint_num*(index+1)]))
    env = make_env(robot_instance=robot_instances, cameras=cameras)
    while True:
        image_list = []  # for visualization
        print('Reset environment...')
        env.reset(sleep_time=1)
        v = input(f'Press Enter to start evaluation or z and Enter to exit...')
        if v == 'z':
            break
        ts = env.reset()
        orientation = [0,0,0]

        init_joint_q =  [
            0.04558632895350456,
            0.001716639962978661,
            0.005531395319849253,
            1.7381932735443115,
            -0.11387044936418533,
            -1.7801556587219238
        ]
        airbot_player.set_target_joint_q(init_joint_q)
        time.sleep(1)
        scale = 1.5

        try:
            #session = requests.session()
            for t in tqdm(range(cfg.time_steps)):
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                curr_image = get_image(ts, camera_names, image_mode).cpu().numpy()
                #curr_image = obs['images']['1']
                curr_image = np.transpose(np.squeeze(curr_image), (1,2,0))
                #curr_image = resize_image(curr_image, (224, 224))
                height, width = curr_image.shape[:2]
                new_size = min(height, width)
                start_x = (width - new_size) // 2
                start_y = (height - new_size) // 2
                curr_image = curr_image[start_y:start_y + new_size, start_x:start_x + new_size]
                curr_image = cv2.resize(curr_image, (224,224))
                curr_image = (curr_image * 255).astype(np.uint8)

                action = get_vla_action(
                    vla=model, cfg=cfg, image = curr_image, processor=processor, task_label=cfg.command,
                    )
                
                current_position, current_orientation = airbot_player.get_current_pose()
                current_end = airbot_player.get_current_end()
                new_position = [
                    current_position[0] + action[0] * scale, # 
                    current_position[1] + action[1] * scale, # 
                    current_position[2] + action[2] * scale, # 
                ]
                orientation[0] += action[3]*scale
                orientation[1] += action[4]*scale
                orientation[2] += action[5]*scale
                new_quaternion = euler_to_quaternion(orientation[0], orientation[1], orientation[2])

                new_end = action[6] # current_end * 0.5 + action[6] * 0.5
                airbot_player.set_target_pose(new_position, new_quaternion, vel=2.0) #vel.2.0
                airbot_player.set_target_end(new_end)

                # # smooth conrtrol
                # current_position, current_orientation = airbot_player.get_current_pose()
                # current_end = airbot_player.get_current_end()
                # if t==0:
                #     action_pre = get_vla_action(
                #     vla=model, cfg=cfg, image = curr_image, processor=processor, task_label=cfg.command,
                #     )
                #     orientation_pre = [0,0,0]
                # action = get_vla_action(
                #     vla=model, cfg=cfg, image = curr_image, processor=processor, task_label=cfg.command,
                #     )
                # gripper = np.array(action[6])
                # pre_gripper = np.array(action_pre[6])
                # new_position = np.array([
                #     current_position[0] + action[0] * scale, # 
                #     current_position[1] + action[1] * scale, # 
                #     current_position[2] + action[2] * scale, # 
                # ])
                # pre_position = np.array([
                #     current_position[0] + action_pre[0] * scale, # 
                #     current_position[1] + action_pre[1] * scale, # 
                #     current_position[2] + action_pre[2] * scale, # 
                # ])
                # for i in range(UPSAMPLE):
                #     orientation[0] += action[3]*scale
                #     orientation[1] += action[4]*scale
                #     orientation[2] += action[5]*scale
                #     orientation = np.array(orientation)
                #     orientation_pre[0] += action_pre[3]*scale
                #     orientation_pre[1] += action_pre[4]*scale
                #     orientation_pre[2] += action_pre[5]*scale
                #     orientation_pre = np.array(orientation_pre)

                #     orientation_new = orientation_pre[:] * (1 - i / UPSAMPLE)+ orientation[:] * (i / UPSAMPLE)
                #     new_quaternion = euler_to_quaternion(orientation_new[0], orientation_new[1], orientation_new[2])
                #     airbot_player.set_target_pose(
                #         pre_position[:] * (1 - i / UPSAMPLE)+ new_position[:] * (i / UPSAMPLE),
                #         new_quaternion,
                #         False,
                #         )
                #     airbot_player.set_target_end(
                #         pre_gripper * (1 - i / UPSAMPLE)
                #         + gripper * (i / UPSAMPLE),
                #         )
                #     time.sleep(DT/UPSAMPLE)
                # orientation_pre = orientation
                # action_pre = action

                #time.sleep(0.3)
                ts = env.step(action=None)
     
        except KeyboardInterrupt as e:
            print(e)
            print('Evaluation interrupted by user...')