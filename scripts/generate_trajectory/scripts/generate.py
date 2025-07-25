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
from utils import process_trajectory, get_corrected_positions, get_gripper_mask, get_bounding_boxes, show_box, mask_to_pos_naive, sq

class NumpyFloatValuesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

if __name__ == '__main__':

    local_path="/home/lixiang/codebase/embodied-CoT/datasets/libero_spatial/libero_spatial/1.0.0"
    builder = tfds.builder_from_directory(builder_dir=local_path)
    ds = builder.as_dataset(split=f'train[{0}%:{100}%]')

    results_path = '/home/lixiang/codebase/embodied-CoT/scripts/generate_trajectory/trajectory_data'
    results_json_path = os.path.join(results_path, "full_trajectory_liberospatial.json")


    results_json = {}
    for i, episode in enumerate(tqdm(ds)):
        # episode_id = i #episode["episode_metadata"]["episode_id"].numpy() ##for bridgev2
        file_path = episode["episode_metadata"]["file_path"].numpy().decode()
        episode_id = episode["episode_metadata"]["episode_id"].numpy()

        plot = True if i%10==0 else False
        gripper_pos_list = get_corrected_positions(episode, episode_id, plot)

        episode_json = {
            "episode_id": int(episode_id),
            "file_path": file_path,
            "gripper_position": gripper_pos_list.tolist(),
        }

        if file_path not in results_json.keys():
            results_json[file_path] = {}

        results_json[file_path][int(episode_id)] = episode_json

        with open(results_json_path, "w") as f:
            json.dump(results_json, f, cls=NumpyFloatValuesEncoder)