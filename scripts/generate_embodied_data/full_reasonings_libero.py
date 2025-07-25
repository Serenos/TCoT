import json
import os
import re
import time

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

from scripts.generate_embodied_data.primitive_movements import get_move_primitives_episode

import tensorflow_datasets as tfds
from prismatic import load # ?important for tfds to not cost GPU memory

from call_llm import call_qianwen
from utils import parse_trajectory_file, replace_same_content, save_trajectory_file

from tqdm import tqdm

def build_prompt2(features, language_instruction, caption=None, list_only_moves=False):
    structured_features = "{\n"

    keys = list(features.keys())

    for i in range(len(features[keys[0]])):
        if list_only_moves:
            structured_features = structured_features + f'    {i}: "{features["move_primitive"][i]}"\n'
        else:
            structured_features = structured_features + f'    {i}: {"{"}\n'

            for key in keys:
                feature_value = features[key][i]
                if isinstance(feature_value, str):
                    feature_value = f'"{feature_value}"'

                structured_features = structured_features + f'        "{key}": {feature_value},\n'

            structured_features = structured_features + "    },\n"

    structured_features = structured_features + "}"

    if list_only_moves:
        features_desc = (
            "Each entry in that dictionary corresponds to a single step on the "
            "trajectory and describes the move that is about to be executed."
        )
    else:
        features_desc = (
            "Each entry in that dictionary corresponds to a single step on "
            "the trajectory. The provided features are the following:\n"
            "\n"
            '- "state_3d" are the current 3d coordinates of the robotic arm end effector; '
            "moving forward increases the first coordinate; moving left increases the second "
            "coordinate; moving up increases the third coordinate,\n"
            '- "move_primitive" describes the move that is about to be executed,\n'
            '- "gripper_position" denotes the location of the gripper in the 256x256 image observation'
        )

    if caption is None:
        caption = ""
    else:
        caption = f"""## Scene description

The robot is operating in the following environment. {caption}

"""

    break_line = ""  # for line formatting

    return f"""# Annotate the training trajectory with reasoning

## Specification of the experimental setup

You're an expert reinforcement learning researcher. You've trained an optimal policy for controlling a robotic arm. The
robot successfully completed a task specified by the instruction: "{language_instruction}". For that purpose, the
robotic arm executed a sequence of actions. Consecutive moves that were executed are the following:


```python
trajectory_features = {structured_features}
```

{features_desc}

{caption}## Your objective

I want you to annotate the given trajectory with reasoning. That is, for each step, I need to know not only {
break_line}which action should be chosen, but importantly what reasoning justifies that action choice. I want you to {
break_line}be descriptive and include all the relevant information available. The reasoning should include the task {
break_line}to complete, the remaining high-level steps, the high-level movements that should be executed and why they {
break_line}are required, the premises that allow inferring the direction of each move, including the locations of {
break_line}relevant objects, possible obstacles or difficulties to avoid, and any other relevant justification.

### Begin by describing the task

Start by giving an overview of the task. Make it more comprehensive than the simple instruction. Include the activity, {
break_line}the objects the robotic arm interacts with, and their relative locations in the environment. Then, describe {
break_line}the high-level movements that were most likely executed, based on the task that was completed and the {
break_line}primitive movements that were executed. Then, for each high-level movement write the interval of steps that {
break_line}movement consists of. Also, for each high-level movement write a justification for why it should be {
break_line}executed. Write an answer for this part using markdown and natural language. Be descriptive and highlight {
break_line}all the relevant details, but ensure that your description is consistent with the trajectory that was {
break_line}executed, specified by the features listed above in the `trajectory_features` dictionary.

### List the reasonings for each step

Finally, for each step describe the reasoning that allows to determine the correct action. For each step describe the {
break_line}remaining part of the objective, the current progress, the objects that are still relevant for determining {
break_line}the plan, and the plan for the next steps, based on the available features. Start the reasoning from a high {
break_line}level and gradually add finer features. I need you to be descriptive and very precise. Ensure that the {
break_line}reasoning is consistent with the task and the executed trajectory. Write the answer for this part as a {
break_line}Python-executable dictionary. For every step in the initial trajectory there should be exactly one separate {
break_line}item of the form <step id>:<reasoning>. <reasoning> can be set to 'same' if it is the same as last step. The final dictionary should have exactly {
break_line}the same set of integer keys as the dictionary of features provided in the `trajectory_features` dictionary {
break_line}above. The reasoning should be a single string that describes the reasoning in natural language and {
break_line}includes all the required features.

Each reasoning string should strictly follow the following form:
- Describe the full task that remains to be completed (but only describe what remains), and place it inside a {
break_line}tag <task>.
- Describe the complete high-level plan for completing the remaining task (the list of remaining high-level steps), {
break_line}and place it inside a tag <plan>.
- Describe the high-level step that should be executed now (chosen from the list of high-level steps), and place it {
break_line}inside a tag <subtask>.
- Describe why the chosen high-level step should be executed now, which features of the current environment influence {
break_line}that decision, and how it should be done. Place it within a tag <subtask_reason>.
- Describe the current primitive movement of the arm that needs to be executed, and place it inside a tag <move>.
- Describe why the chosen movement should be executed now and which features of the current environment influence that {
break_line}decision. Place it inside a tag <move_reason>.

## Task summary

Here is a breakdown of what needs to be done:

- Describe the task.
- Describe the high-level movements that were executed, based on the completed task and the listed features.
- Describe the plan for the solution that allowed the robot to complete the task successfully.
- For each step on the trajectory, describe the reasoning that leads to determining the correct action. {
break_line}Reasoning can be set to 'same' if it is the same as last step. {
break_line}If the reasoning is not 'same', it should exactly contain the following 6 tags: <task>, <plan>, <subtask>, <subtask_reason>, <move>, <move_reason>.
- At the very end of the response, write a single label FINISHED to indicate that the answer is complete."""

def find_task_occurrences(input_string, tags):
    pattern = r"(\d+):"
    for tag in tags:
        pattern = pattern + r"\s*<" + tag + r">([^<]*)<\/" + tag + ">"

    matches = re.findall(pattern, input_string)
    return matches


def extract_reasoning_dict(reasoning_output, tags=("task", "plan", "subtask", "subtask_reason", "move", "move_reason")):
    if reasoning_output is None:
        return dict()

    trajectory = dict()

    matches = find_task_occurrences(reasoning_output, tags)

    for match in matches:
        trajectory[int(match[0])] = dict(zip(tags, match[1:]))

    return trajectory


def get_reasoning_dict(features, metadata, episode_id):
    language_instruction = metadata["language_instruction"]
    caption = metadata["caption"] if "caption" in metadata.keys() else None

    prompt_pth =       f'/home/lixiang/codebase/embodied-CoT/scripts/generate_embodied_data/generated_data/libero_10/prompt/{episode_id}.txt'
    input_file_path =  f'/home/lixiang/codebase/embodied-CoT/scripts/generate_embodied_data/generated_data/libero_10/prompt_answer/{episode_id}.txt'
    output_file_path = f'/home/lixiang/codebase/embodied-CoT/scripts/generate_embodied_data/generated_data/libero_10/prompt_answer_processed/{episode_id}_new.txt'  # replace with your desired output file path

    if not os.path.exists(output_file_path):
        prompt = build_prompt2(features, language_instruction, caption=caption, list_only_moves=True)
        #print("metadata:", metadata, "\nprompt:", prompt)

        # call quanwen vl and save answer to txt file
        reasoning_output = call_qianwen(prompt, episode_id, prompt_pth, input_file_path) 
        trajectory_data = parse_trajectory_file(input_file_path)
        # Replace "same" entries
        updated_trajectory_data = replace_same_content(trajectory_data)
        # Save the updated data to a new file
        save_trajectory_file(updated_trajectory_data, output_file_path)

    with open(output_file_path, 'r') as f:
        reasoning_output = f.read()

    #print("reasoning:", reasoning_output)

    return extract_reasoning_dict(reasoning_output)


def build_single_reasoning(episode_id, ds, captions):
    # ds = builder.as_dataset(split=f"train[{episode_id}:{episode_id + 1}]")
    # episode = next(iter(ds))
    for i, episode in enumerate(ds):
        if i == episode_id:
            episode_id = i #episode["episode_metadata"]["episode_id"].numpy()
            #file_path = episode["episode_metadata"]["file_path"].numpy().decode()
            break
        
    ft = dict()

    ft["state_3d"] = [list(step["observation"]["state"][:3].numpy()) for step in episode["steps"]]

    move_primitives = get_move_primitives_episode(episode)
    ft["move_primitive"] = [move[0] for move in move_primitives]

    mt = {
        "episode_id": str(episode_id),
        "file_path": str(episode["episode_metadata"]["file_path"].numpy())[2:-1],
        "n_steps": len(episode["steps"]),
        "language_instruction": str(next(iter(episode["steps"]))["language_instruction"].numpy().decode()),
    }

    mt["caption"] = captions[mt["file_path"]][mt["episode_id"]]["caption"]

    reasoning = get_reasoning_dict(ft, mt, episode_id)
    entry = {"reasoning": reasoning, "features": ft, "metadata": mt}

    return entry


def generate_reasonings(builder, episode_ids, save_path="full_reasonings_libero_10.json"):
    reasonings = dict()

    if os.path.exists(save_path):
        print(save_path, "existing, loading contents")
        with open(save_path, "r") as f:
            reasonings = json.load(f)

        print("loaded reasonings:", sum([len(v) for v in reasonings.values()]), "entries")

    with open("/home/lixiang/codebase/embodied-CoT/scripts/generate_embodied_data/generated_data/full_description_libero_10.json", "r") as captions_file:
        captions_dict = json.load(captions_file)

    ds = builder.as_dataset(split=f'train[{0}%:{100}%]')
    for i in tqdm(episode_ids):
        entry = build_single_reasoning(i, ds, captions_dict)

        if entry["metadata"]["file_path"] in reasonings.keys():
            reasonings[entry["metadata"]["file_path"]][entry["metadata"]["episode_id"]] = entry
        else:
            reasonings[entry["metadata"]["file_path"]] = {entry["metadata"]["episode_id"]: entry}

        #print("computed reasoning:", entry)
    import numpy as np
    def custom_serializer(obj):
        if isinstance(obj, np.float32):
            return float(obj)
        raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')
    with open(save_path, "w") as out_f:
        json.dump(reasonings, out_f, default=custom_serializer)


if __name__ == '__main__':

    local_path="/home/lixiang/codebase/embodied-CoT/datasets/libero/libero_10_no_noops/1.0.0"
    builder = tfds.builder_from_directory(builder_dir=local_path)
    #ds = b.as_dataset(split=f'train[{start}%:{end}%]')
    generate_reasonings(builder=builder, episode_ids=range(0,379)) #(0, 432) for libero-spatial dataset; (0, 379) for libero-10