import argparse
import json
import os
import time
import warnings

import tensorflow_datasets as tfds
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from utils import NumpyFloatValuesEncoder, post_process_caption

from prismatic import load # ?important for tfds to not cost GPU memory
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()

# parser.add_argument("--id", default=0, type=int)
# parser.add_argument("--gpu", default=0, type=int)
# parser.add_argument("--splits", default=1, type=int)
parser.add_argument("--data-path", type=str)
parser.add_argument("--result-path", default="/home/lixiang/codebase/embodied-CoT/scripts/generate_embodied_data/generated_data")

args = parser.parse_args()
bbox_json_path = os.path.join(args.result_path, f"full_bboxes_libero_10.json")

print("Loading data...")
# split_percents = 100 // args.splits
start = 0 #args.id * split_percents
end = 100 # (args.id + 1) * split_percents

# ds = tfds.load("bridge_orig", data_dir=args.data_path, split=f"train[{start}%:{end}%]")
local_path="/home/lixiang/codebase/embodied-CoT/datasets/libero/libero_10_no_noops/1.0.0"
b = tfds.builder_from_directory(builder_dir=local_path)
ds = b.as_dataset(split=f'train[{start}%:{end}%]')
print("Done.")

print("Loading Prismatic descriptions...")
results_json_path = "/home/lixiang/codebase/embodied-CoT/scripts/generate_embodied_data/generated_data/full_description_libero_10.json"
with open(results_json_path, "r") as f:
    results_json = json.load(f)
print("Done.")

device = "cuda:0"
print(f"Loading gDINO to device {device}...")
model_id = "IDEA-Research/grounding-dino-base"


processor = AutoProcessor.from_pretrained(model_id, size={"shortest_edge": 256, "longest_edge": 256})
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
print("Done.")

BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.2

bbox_results_json = {}
for ep_idx, episode in enumerate(ds):

    #episode_id = episode["episode_metadata"]["episode_id"].numpy()
    episode_id = ep_idx
    file_path = episode["episode_metadata"]["file_path"].numpy().decode()
    print(f"starting ep: {episode_id}, {file_path}")

    if file_path not in bbox_results_json.keys():
        bbox_results_json[file_path] = {}

    episode_json = results_json[file_path][str(episode_id)]
    description = episode_json["caption"]

    start = time.time()
    bboxes_list = []
    for step_idx, step in enumerate(episode["steps"]):
        if step_idx == 0:
            lang_instruction = step["language_instruction"].numpy().decode()
        #image = Image.fromarray(step["observation"]["image_0"].numpy())
        image = Image.fromarray(step["observation"]["image"].numpy()) #for libero dataset
        inputs = processor(
            images=image,
            text=post_process_caption(description, lang_instruction),
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[image.size[::-1]],
        )[0]

        logits, phrases, boxes = (
            results["scores"].cpu().numpy(),
            results["labels"],
            results["boxes"].cpu().numpy(),
        )

        bboxes = []
        for lg, p, b in zip(logits, phrases, boxes):
            b = list(b.astype(int))
            lg = round(lg, 5)
            bboxes.append((lg, p, b))
            break

        bboxes_list.append(bboxes)
        # break
    end = time.time()
    bbox_results_json[file_path][str(ep_idx)] = {
        "episode_id": int(episode_id),
        "file_path": file_path,
        "bboxes": bboxes_list,
    }

    with open(bbox_json_path, "w") as f:
        json.dump(bbox_results_json, f, cls=NumpyFloatValuesEncoder)
    print(f"finished ep ({ep_idx} / {len(ds)}). Elapsed time: {round(end - start, 2)}")
