import tensorflow_datasets as tfds
from prismatic import load # ?important for tfds to not cost GPU memory
from sklearn.linear_model import RANSACRegressor
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import os
import json
import numpy as np
from tqdm import tqdm
import torch

suit = 'liber_o10' #libero_spatial, libero_goal, libero_object, liber_o10
device = 'cuda:1'
cotracker = torch.hub.load(os.path.join(os.path.expanduser("~"), ".cache/torch/hub/facebookresearch_co-tracker_main/"), "cotracker3_offline", source="local").to(device)


def as_gif(images, episode_id):
    # Render the images as the gif:
    path = f'/home/lixiang/codebase/embodied-CoT/scripts/generate_trajectory/trajectory_vis3/{suit}/{suit}_{episode_id}.gif'
    images[0].save(path, save_all=True, append_images=images[1:], duration=200, loop=0)
    gif_bytes = open(path,'rb').read()
    return gif_bytes

def get_trajectory_by_cotracker(episode, episode_id, plot=False):
    images = [step["observation"]['image'].numpy() for step in episode["steps"]]
    images_tensor = [torch.from_numpy(tensor) for tensor in images]
    frame_tensor = torch.stack(images_tensor, dim=0) ## T H W C
    frame_tensor = frame_tensor.permute(0, 3, 1, 2)[None].float().to(device) ## B T C H W
    #print(frame_tensor.shape)
    # it's better to track together
    queries = torch.tensor([
        [0., 128., 45.],  # point tracked from the first frame
        [0., 128., 50.],
        [0., 128., 40.],
        [0., 123., 45.],
        [0., 133., 45.],
    ]).to(device)
    #print(queries.shape)
    pred_tracks, pred_visibility = cotracker(frame_tensor, queries=queries[None])
    track_points = pred_tracks[0,:,0,:].cpu().numpy()
    
    if plot:
        for i, image in enumerate(images):
            for p in track_points[i:]:
                cv2.circle(image, (int(p[0]), int(p[1])), radius=2, color=(0, 0, 255), thickness=-1)
        image_plt = [Image.fromarray(image) for image in images]
        as_gif(image_plt, episode_id)
    #print(pred_tracks.shape)
    return track_points

class NumpyFloatValuesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

if __name__ == '__main__':

    local_path=f"/home/lixiang/codebase/embodied-CoT/datasets/libero_new/{suit}/1.0.0"
    builder = tfds.builder_from_directory(builder_dir=local_path)
    ds = builder.as_dataset(split=f'train[{0}%:{100}%]')

    results_path = '/home/lixiang/codebase/embodied-CoT/scripts/generate_trajectory/trajectory_data'
    results_json_path = os.path.join(results_path, f"full_trajectory_v3_{suit}.json")


    results_json = {}
    for i, episode in enumerate(tqdm(ds)):
        # episode_id = i #episode["episode_metadata"]["episode_id"].numpy() ##for bridgev2
        file_path = episode["episode_metadata"]["file_path"].numpy().decode()
        episode_id = episode["episode_metadata"]["episode_id"].numpy()

        plot = True if i%2==0 else False
        trajectory_points = get_trajectory_by_cotracker(episode, episode_id, plot)

        episode_json = {
            "episode_id": int(episode_id),
            "file_path": file_path,
            "trajectory_points": trajectory_points.tolist(),
        }

        if file_path not in results_json.keys():
            results_json[file_path] = {}

        results_json[file_path][int(episode_id)] = episode_json

        with open(results_json_path, "w") as f:
            json.dump(results_json, f, cls=NumpyFloatValuesEncoder)