import argparse
import json
import os
import warnings

import tensorflow_datasets as tfds
import torch
from PIL import Image
from tqdm import tqdm
from utils import NumpyFloatValuesEncoder

from prismatic import load

parser = argparse.ArgumentParser()

# parser.add_argument("--id", type=int)
# parser.add_argument("--gpu", type=int)
# parser.add_argument("--splits", default=4, type=int)
parser.add_argument("--results-path", default="/home/lixiang/codebase/embodied-CoT/scripts/generate_embodied_data/generated_data")

args = parser.parse_args()

device = f"cuda:0"
hf_token = "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" #"<TODO: Insert HF Token>"
vlm_model_id = "prism-dinosiglip+7b"

warnings.filterwarnings("ignore")


#split_percents = 100 // args.splits
start = 0 #args.id * split_percents
end = 100 #(args.id + 1) * split_percents

# Load Bridge V2
# ds = tfds.load(
#     "bridge_orig",
#     data_dir="<TODO: Enter path to BridgeV2>",
#     split=f"train[{start}%:{end}%]",
# )

local_path="/home/lixiang/codebase/embodied-CoT/datasets/libero/libero_10_no_noops/1.0.0"
b = tfds.builder_from_directory(builder_dir=local_path)
ds = b.as_dataset(split=f'train[{start}%:{end}%]')

# Load Prismatic VLM
print(f"Loading Prismatic VLM ({vlm_model_id})...")
vlm = load(vlm_model_id, hf_token=hf_token)
vlm = vlm.to(device, dtype=torch.bfloat16)

results_json_path = os.path.join(args.results_path, f"full_description_libero_10.json")


def create_user_prompt(lang_instruction):
    user_prompt = "Briefly describe the things in this scene and their spatial relations to each other."
    # user_prompt = "Briefly describe the objects in this scene."]
    lang_instruction = lang_instruction.strip()
    if len(lang_instruction) > 0 and lang_instruction[-1] == ".":
        lang_instruction = lang_instruction[:-1]
    if len(lang_instruction) > 0 and " " in lang_instruction:
        user_prompt = f"The robot task is: '{lang_instruction}.' " + user_prompt
    return user_prompt


results_json = {}
for i, episode in enumerate(tqdm(ds)):
    episode_id = i #episode["episode_metadata"]["episode_id"].numpy()
    file_path = episode["episode_metadata"]["file_path"].numpy().decode()
    for step in episode["steps"]:
        lang_instruction = step["language_instruction"].numpy().decode()
        #image = Image.fromarray(step["observation"]["image_0"].numpy()) #for bridgev2
        image = Image.fromarray(step["observation"]["image"].numpy()) #for libero dataset

        # user_prompt = "Describe the objects in this scene. Be specific."
        user_prompt = create_user_prompt(lang_instruction)
        prompt_builder = vlm.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=user_prompt)
        prompt_text = prompt_builder.get_prompt()

        torch.manual_seed(0)
        caption = vlm.generate(
            image,
            prompt_text,
            do_sample=True,
            temperature=0.4,
            max_new_tokens=64,
            min_length=1,
        )
        break

    episode_json = {
        "episode_id": int(episode_id),
        "file_path": file_path,
        "caption": caption,
    }

    if file_path not in results_json.keys():
        results_json[file_path] = {}

    results_json[file_path][int(episode_id)] = episode_json

    with open(results_json_path, "w") as f:
        json.dump(results_json, f, cls=NumpyFloatValuesEncoder)
