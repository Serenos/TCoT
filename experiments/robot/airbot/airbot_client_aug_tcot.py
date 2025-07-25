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
import matplotlib.pyplot as plt

def get_action(image=np.zeros((224,224,3), dtype=np.uint8), command='do something', unnorm_key='bridge_orig', traj_type='global', reset_frozen=True, session=None):
    # unnorm_key should be changed accordingly
    data = session.post(
        "http://localhost:8000/act",
        json={"image": image, "instruction": command, "unnorm_key" : unnorm_key, "traj_type": traj_type, "reset_frozen": reset_frozen}
    ).json()
    action = data.get("action")
    trajectory = data.get("trajectory")
    return action, trajectory

# Function to convert roll, pitch, yaw to quaternion
def euler_to_quaternion(roll, pitch, yaw):
    r = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=False)
    return r.as_quat()

import os
import imageio
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
def save_rollout_video(rollout_images, idx, success, task_description, note=None):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"./rollouts_real/{note}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=15)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")

    return mp4_path


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

import tensorflow as tf
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

def draw_traj(img, generated_txt, horizon, traj_type):
    #print('generated_txt: ', generated_txt)
    str_traj = generated_txt.split("Trajectory: ")[-1].split("Action: ")[0]
    #print('str_traj: ', str_traj)
    if str_traj[0] != '[':
        str_traj = '['+str_traj
    if traj_type == 'global' or traj_type == 'local':
        dim=2 #[x,y]
    else:
        dim=4 #[x,y,depth,gripper]
    traj = json.loads(str_traj)
    traj = traj[:dim*horizon]
    x = np.array(traj[::dim]) * 224 // 256
    y = np.array(traj[1::dim]) * 224 // 256
    if traj_type == 'globalv4' or traj_type == 'localv4':
        depth = np.array(traj[2::dim])
        gripper_state = np.array(traj[3::dim])

    num_point = min(len(x), len(y))
    cmap = plt.get_cmap('plasma')  # 可以选择其他的颜色映射，例如 'plasma', 'inferno', 'magma', 'cividis'  'viridis'等
    for i in range(num_point):
        # 计算当前点的颜色
        if traj_type == 'globalv4':
            color = cmap(depth[i]/255)  # 归一化到 [0, 1] 范围
        else:
            color = cmap(i / (num_point - 1))  # 归一化到 [0, 1] 范围
        color = tuple(int(c * 255) for c in color[:3])  # 将颜色转换为 BGR 格式，并缩放到 [0, 255]
        cv2.circle(img, (int(x[i]), int(y[i])), radius=2, color=color, thickness=-1)
    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-cmd', "--command", action='store', type=str, help='command to do', default="Do something.", required=False)
    parser.add_argument('-uk', "--unnorm_key", action='store', type=str, help='unnorm key', default="bridge_orig", required=False)
    parser.add_argument('-cw', "--camera_web", action='store', type=str, help='web_camera_indices', default="", required=False)
    parser.add_argument('-cm', "--camera_mv", action='store', type=str, help='machine_vision_camera_indices', default="", required=False)
    parser.add_argument('-can', "--can_buses", action='store', type=str, help='can_bus', default="can0", required=False)
    parser.add_argument('-em', "--eef_mode", action='store', type=str, help='eef_mode', default="gripper")
    parser.add_argument("--time_steps", action='store', type=int, help='max_timesteps', default=20)
    parser.add_argument('-note', "--run_note", action='store', type=str, help='run_note', default="openvla_try1")

    config = vars(parser.parse_args())
    
    camera_names = ['1']
    camera_web = config['camera_web']
    camera_mv = config['camera_mv']
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
    
    # init robots
    robot_instances:List[AssembledRobot] = []

    # modify the path to the airbot_play urdf file
    vel = 2.0
    fps = 30
    joint_num = 7
    robot_num = 1
    start_joint = [0] * 7
    can_list = config['can_buses'].split(',')
    eef_mode = config['eef_mode']
    image_mode = 0
    for index, can in enumerate(can_list):
        airbot_player = airbot.create_agent(direction="down", can_interface="can1", end_mode="gripper") 
        robot_instances.append(AssembledRobot(airbot_player, 1/fps, 
                                                start_joint[joint_num*index:joint_num*(index+1)]))
    env = make_env(robot_instance=robot_instances, cameras=cameras)
    
    # TCoT parameters
    strategy = 'tcot' # [vla, tcot]
    #[None, global, local, globalv4, localv4]
    first_traj_type = 'global' # planA
    second_traj_type = 'local'  # planB
    traj_type = first_traj_type # currenet plan

    # check norm config
    print(f'#############################  config: {config}  ####################################')

    episodes_idx = 0
    replay_images = []
    while True:
        print('Reset environment...')
        traj_type = first_traj_type
        env.reset(sleep_time=1)
        v = input(f'Press Enter to start evaluation or z and Enter to exit... c to change TCoT planning type,,,,\n')
        if v == 'z':
            break
        if v == 'c':
            print(f'changing TCoT planning type from {first_traj_type} to {second_traj_type}')
            traj_type = second_traj_type
        ts = env.reset()
        orientation = [0,0,0]
        print(f'trajectory_type: {traj_type}')
        if episodes_idx > 0:
            # Save a replay video of the episode
            if traj_type==first_traj_type:
                save_rollout_video(
                    replay_images, episodes_idx, success='unknown', task_description=config['command'], note=config['run_note']
                )
                replay_images = []
        if traj_type==first_traj_type:
            episodes_idx += 1
        try:
            session = requests.session()
            reset_frozen = True

            for t in tqdm(range(config['time_steps'])):
                obs = ts.observation
                curr_image = get_image(ts, camera_names, image_mode).cpu().numpy()
                curr_image = np.transpose(np.squeeze(curr_image), (1,2,0))
                height, width = curr_image.shape[:2]
                curr_image = (curr_image * 255).astype(np.uint8)
                # cv2.imwrite('current_img_before.jpg', curr_image)
                curr_image = resize_image(curr_image, (224,224))
                # cv2.imwrite('current_img_mid.jpg', curr_image)
                center_crop = True
                if center_crop:
                    batch_size = 1
                    crop_scale = 0.9

                    # Convert to TF Tensor and record original data type (should be tf.uint8)
                    image = tf.convert_to_tensor(np.array(curr_image))
                    orig_dtype = image.dtype

                    # Convert to data type tf.float32 and values between [0,1]
                    image = tf.image.convert_image_dtype(image, tf.float32)

                    # Crop and then resize back to original size
                    image = crop_and_resize(image, crop_scale, batch_size)

                    # Convert back to original data type
                    image = tf.clip_by_value(image, 0, 1)
                    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

                    # Convert back to PIL Image
                    # image = Image.fromarray(image.numpy())
                    # image = image.convert("RGB")
                    curr_image = image.numpy()
                # cv2.imwrite('current_img_after.jpg', curr_image)
                action, trajectory = get_action(image = curr_image, command = config['command'], unnorm_key = config['unnorm_key'], traj_type=traj_type, reset_frozen=reset_frozen, session=session)
                scale = 1.0
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

                #new_end = action[6] # current_end * 0.5 + action[6] * 0.5
                new_end = 0 if action[6] < 0.6 else 1
                print(f'gripper action: {action[6]} -> {new_end}')
                airbot_player.set_target_pose(new_position, new_quaternion, vel=1.0)
                airbot_player.set_target_end(new_end)
                time.sleep(0.5)
                ts = env.step(action=None)

                reset_frozen = False
                if traj_type == 'local':
                    curr_image = curr_image[:,:,::-1].copy()
                    curr_image = draw_traj(curr_image, trajectory, horizon=10, traj_type=traj_type) #5
                elif traj_type == 'global':
                    curr_image = curr_image[:,:,::-1].copy()
                    curr_image = draw_traj(curr_image, trajectory, horizon=30, traj_type=traj_type)
                replay_images.append(curr_image)

        except KeyboardInterrupt as e:
            print(e)
            print('Evaluation interrupted by user...')