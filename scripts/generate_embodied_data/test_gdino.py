import requests
import tensorflow_datasets as tfds
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import argparse
import json
import os
import time
import warnings
from prismatic import load

# import tensorflow as tf
#from utils import NumpyFloatValuesEncoder, post_process_caption

warnings.filterwarnings("ignore")

model_id = "IDEA-Research/grounding-dino-base" #"IDEA-Research/grounding-dino-tiny"
device = "cuda:0"

split_percents = 100
start = 0
end = 100

local_path="/home/lixiang/codebase/embodied-CoT/datasets/libero/libero_spatial_no_noops/1.0.0"
b = tfds.builder_from_directory(builder_dir=local_path)
ds = b.as_dataset(split=f'train[{start}%:{end}%]')

print("loading model")
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
print("done")



image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)
# Check for cats and remote controls
text = "a cat. a remote control."
text2 = "A black bowl is next to a cookie box, and a plate is nearby. The robot should pick up the black bowl and place it on the plate."
image2 = Image.open('/home/lixiang/codebase/embodied-CoT/scripts/generate_embodied_data/bounding_boxes/output.png')

inputs = processor(images=image, text=text, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.4,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]]
)
print(results)


inputs = processor(images=image2, text=text2, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.4,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]]
)
print(results)


