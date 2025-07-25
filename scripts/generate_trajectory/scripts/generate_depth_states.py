
import tensorflow_datasets as tfds
from prismatic import load # ?important for tfds to not cost GPU memory
from sklearn.linear_model import RANSACRegressor
import cv2
import matplotlib
import mediapy
from IPython import display
from PIL import Image
from matplotlib import pyplot as plt

from transformers import pipeline
from PIL import Image
import tqdm

from tqdm import tqdm  
import os
import json
import numpy as np
from sklearn.linear_model import RANSACRegressor

def as_gif(images, path):
  # Render the images as the gif:
  images[0].save(path, save_all=True, append_images=images[1:], duration=100, loop=0)


def draw_traj(img, traj):
    #print('generated_txt: ', generated_txt)

    x = traj[:,0] # 轨迹的 x 坐标
    y = traj[:,1]                # 轨迹的 y 坐标
    depth = traj[:,2]  # 每个点的深度值 (随机生成)

    # 获取颜色映射
    cmap = plt.get_cmap('plasma')  # 可以选择其他的颜色映射，例如 'plasma', 'inferno', 'magma', 'cividis'  'viridis'等

    for i in range(traj.shape[0]):
        # 计算当前点的颜色
        color = cmap(depth[i] / 255)  # 归一化到 [0, 1] 范围
        color = tuple(int(c * 255) for c in color[:3])  # 将颜色转换为 BGR 格式，并缩放到 [0, 255]
        cv2.circle(img, (int(x[i]), int(y[i])), radius=2, color=color, thickness=-1)
    return img


def norm(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    # 归一化到 [0, 1]
    normalized_arr = (arr - arr_min) / (arr_max - arr_min)
    return normalized_arr

class NumpyFloatValuesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.uint8):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

def generate_depth_map():
    suit = 'libero_spatial'
    device='cuda:3'

    local_path=f"/home/lixiang/codebase/embodied-CoT/datasets/libero_new/{suit}/1.0.0"
    builder = tfds.builder_from_directory(builder_dir=local_path)
    ds = builder.as_dataset(split=f'train[{0}%:{100}%]')

    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf", device=device)

    episode_ids = range(300, 432) #(0,432) for spatial, 

    for idx in tqdm(episode_ids):

        for i, episode in tqdm(enumerate(ds)):
            episode_id = episode["episode_metadata"]["episode_id"].numpy()
            filename = episode["episode_metadata"]["file_path"].numpy().decode()
            if idx == episode_id:
                break
        #print(f'{i}: {episode_id}')
        print('processing......', episode_id)

        images = [step["observation"]['image'] for step in episode["steps"]]
        images = [Image.fromarray(image.numpy()) for image in images]
        states = [step['observation']['state'] for step in episode['steps']]
        actions = [step['action'] for step in episode['steps']]
        #print(len(images))

        depth_list = []
        for image in images:    
            depth = pipe(image)["depth"]
            depth_list.append(depth)
        
        as_gif(images=depth_list, path=f'/home/lixiang/codebase/embodied-CoT/scripts/generate_trajectory/trajectory_depth/{suit}/depth_{episode_id}.gif')

def add_depth_to_trajectory(vis=False):
    suit = 'libero_spatial'
    device='cuda:3'
    depth_path = f'/home/lixiang/codebase/embodied-CoT/scripts/generate_trajectory/trajectory_depth/{suit}'

    results_path = '/home/lixiang/codebase/embodied-CoT/scripts/generate_trajectory/trajectory_data'
    results_json_path = os.path.join(results_path, f"full_trajectory_v3_{suit}.json")

    depth_results_json_path = os.path.join(results_path, f"full_trajectory_v4_{suit}.json")

    local_path=f"/home/lixiang/codebase/embodied-CoT/datasets/libero_new/{suit}/1.0.0"
    builder = tfds.builder_from_directory(builder_dir=local_path)
    ds = builder.as_dataset(split=f'train[{0}%:{100}%]')

    with open(results_json_path, "r") as f:
        trajs = json.load(f)
    
    episode_ids = range(0,432) #(0,432) for spatial, 
    for idx in tqdm(episode_ids):

        for i, episode in enumerate(ds):
            episode_id = episode["episode_metadata"]["episode_id"].numpy()
            filename = episode["episode_metadata"]["file_path"].numpy().decode()
            if idx == episode_id:
                break
        #print(f'{i}: {episode_id}')
        print('processing......', episode_id)
        trajectory_points = trajs[filename][f'{idx}']['trajectory_points'] #[x,y] (N,2)

        actions = [step['action'] for step in episode['steps']]

        depth_list = []
        with Image.open(f'{depth_path}/depth_{episode_id}.gif') as im:
            # 获取 GIF 的总帧数
            frame_count = im.n_frames
            print('gif frame_count: ', frame_count)
            # 遍历每一帧
            for i in range(frame_count):
                # 寻求到当前帧
                im.seek(i)
                img = im.convert('L')
                img_array = np.array(img)
                depth_list.append(img_array)
        
        traj_depth_points = []
        states = [step['observation']['state'] for step in episode['steps']]
        states = np.array(states)
        gripper_norm_states = norm(states[:,-1]) #noralize to [0,1]
        for i in range(len(trajectory_points)):
            point = trajectory_points[i]
            depth_img = depth_list[i]
            point_depth = depth_img[int(point[0]), int(point[1])] #[0,255]
            point.append(point_depth)
            point.append(gripper_norm_states[i])
            traj_depth_points.append(point)
        print('origin traj_depth_points shape: ', np.array(traj_depth_points).shape)

        #fit the discrete depth with RANSAC
        traj_depth_points_numpy = np.array(traj_depth_points)
        points_2d = np.array(traj_depth_points_numpy[:,:3], dtype=np.float32)
        points_3d = np.array(states[:,:3], dtype=np.float32)
        points_3d_pr = np.concatenate([points_3d, np.ones_like(points_3d[:, :1])], axis=-1)
        points_2d_pr = np.concatenate([points_2d, np.ones_like(points_2d[:, :1])], axis=-1)
        reg = RANSACRegressor(random_state=0).fit(points_3d_pr, points_2d_pr)
        pr_pos = reg.predict(points_3d_pr)[:, :-1].astype(int)
        print('fit depth trajectory shape: ', pr_pos.shape)
        traj_depth_points_numpy[:, 2] = pr_pos[:, 2]
        traj_depth_points = traj_depth_points_numpy.tolist()

        if vis:
            images = [step["observation"]["image"] for step in episode["steps"]]
            images = [img.numpy() for img in images]
            for i, image in enumerate(images):
                image = draw_traj(image, np.array(traj_depth_points))

            image_plt = [Image.fromarray(image) for image in images]
            as_gif(images=image_plt, path=f'/home/lixiang/codebase/embodied-CoT/scripts/generate_trajectory/traj_depth_vis/smooth_depth_trajectory_{episode_id}.gif')


        trajs[filename][f'{idx}']['trajectory_points'] = traj_depth_points
        with open(depth_results_json_path, "w") as f:
            json.dump(trajs, f, cls=NumpyFloatValuesEncoder)

add_depth_to_trajectory(vis=False)