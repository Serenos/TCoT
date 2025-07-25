"""
deploy.py

Provide a lightweight server/client implementation for deploying OpenVLA models (through the HF AutoClass API) over a
REST API. This script implements *just* the server, with specific dependencies and instructions below.

Note that for the *client*, usage just requires numpy/json-numpy, and requests; example usage below!

Dependencies:
    => Server (runs OpenVLA model on GPU): `pip install uvicorn fastapi json-numpy`
    => Client: `pip install requests json-numpy`

Client (Standalone) Usage (assuming a server running on 0.0.0.0:8000):

```
import requests
import json_numpy
json_numpy.patch()
import numpy as np

action = requests.post(
    "http://0.0.0.0:8000/act",
    json={"image": np.zeros((256, 256, 3), dtype=np.uint8), "instruction": "do something"}
).json()

Note that if your server is not accessible on the open web, you can use ngrok, or forward ports to your client via ssh:
    => `ssh -L 8000:localhost:8000 ssh USER@<SERVER_IP>`
"""

import os.path

# ruff: noqa: E402
import json_numpy

json_numpy.patch()
import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import draccus
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers import AutoConfig, AutoImageProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import  OpenVLAForActionPrediction, TcotSearchForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from transformers import AutoModelForVision2Seq

AutoConfig.register("openvla", OpenVLAConfig)
AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
#AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
AutoModelForVision2Seq.register(OpenVLAConfig, TcotSearchForActionPrediction)

from peft import LoraConfig, PeftModel
# === Utilities ===
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_openvla_prompt(instruction: str, traj: str) -> str:
    if traj == 'global':
        prompt = f"In: What action should the robot take to {instruction.lower()}? Please predict the gripper global trajectory and the action.\nOut: "
    elif traj == 'local':
        prompt = f"In: What action should the robot take to {instruction.lower()}? Please predict the  gripper local trajectory and the action.\nOut: "
    elif traj == 'globalv4':
        prompt = f"In: What action should the robot take to {instruction.lower()}? Please predict the  gripper global trajectory (x,y,depth,gripper_state) and the action.\nOut: "
    else:
        prompt = f"In: What action should the robot take to {instruction.lower()}? Please predict the gripper trajectory and the action.\nOut: "
    return prompt


# === Server Interface ===
class OpenVLAServer:
    def __init__(self, openvla_path: Union[str, Path], attn_implementation: Optional[str] = "flash_attention_2") -> Path:
        """
        A simple server for OpenVLA models; exposes `/act` to predict an action for a given image + instruction.
            => Takes in {"image": np.ndarray, "instruction": str, "unnorm_key": Optional[str]}
            => Returns  {"action": np.ndarray}
        """
        self.openvla_path, self.attn_implementation = openvla_path, attn_implementation
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        # Load VLA Model using HF AutoClasses
        self.processor = AutoProcessor.from_pretrained(self.openvla_path, trust_remote_code=True)
        # self.vla = AutoModelForVision2Seq.from_pretrained(
        #     self.openvla_path,
        #     attn_implementation=attn_implementation,
        #     torch_dtype=torch.bfloat16,
        #     low_cpu_mem_usage=True,
        #     trust_remote_code=True,
        # ).to(self.device)


        using_lora = True
        if using_lora:
            adapter_dir = "/home/cvailab/codebase/embodied-CoT/openvla/exp_model/lora_model_airbot/openvla-7b+airbot_feed+tcot+airbot_feed_with_spoon+cofintune0.3/step_4000"
            print(f'loading lora model from {adapter_dir} for inference, base model from {self.openvla_path}')
            
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, TcotSearchForActionPrediction)

        base_vla = AutoModelForVision2Seq.from_pretrained(
            self.openvla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True, attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
        )
        if using_lora:
            model = PeftModel.from_pretrained(base_vla, adapter_dir)
        else:
            model = base_vla #if not using lora adaptor
        
        self.vla = model.to(self.device)

        # [Hacky] Load Dataset Statistics from Disk (if passing a path to a fine-tuned model)
        if os.path.isdir(self.openvla_path):
            with open(Path(self.openvla_path) / "dataset_statistics.json", "r") as f:
                self.vla.norm_stats = json.load(f)
                print(f'loading norm_statas from {self.openvla_path} dataset_statistics.json.......')

    def predict_action(self, payload: Dict[str, Any]) -> str:
        try:
            if double_encode := "encoded" in payload:
                # Support cases where `json_numpy` is hard to install, and numpy arrays are "double-encoded" as strings
                assert len(payload.keys() == 1), "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            # Parse payload components
            image, instruction = payload["image"], payload["instruction"]
            unnorm_key = payload.get("unnorm_key", None)
            traj_type = payload.get("traj_type")
            reset_frozen = payload.get("reset_frozen")

            # Run VLA Inference
            prompt = get_openvla_prompt(instruction, traj_type)
            # reset frozen paraemters every episode
            if reset_frozen:
                if traj_type == 'local':
                    self.vla.max_freezing_time = 10
                else:
                    self.vla.max_freezing_time = 100
                print(f'###################################################\nreset_frozen, self.vla.max_freezing_time: {self.vla.max_freezing_time}')
                self.vla.time_frozen = 0
            # print(f'model.max_freezing_time: {model.max_freezing_time}')
            # re-use previous trajectory for frozen times
            if self.vla.freezing_cot:
                if self.vla.time_frozen <= 0:
                    self.vla.frozen_prompt = self.vla.base_prompt
                    self.vla.time_frozen = self.vla.max_freezing_time
                self.vla.time_frozen -= 1
                prompt = prompt + self.vla.frozen_prompt


            inputs = self.processor(prompt, Image.fromarray(image).convert("RGB")).to(self.device, dtype=torch.bfloat16)
            #action, _ = self.vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)


            torch.manual_seed(0)
            action, generated_ids, generated_dicts = self.vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False, max_new_tokens=512, return_dict_in_generate=True, output_scores=True)

                
            generated_text = self.processor.batch_decode(generated_ids[:,1:-1])[0] #remove <s> </s>

            if self.vla.freezing_cot:
                prompt = generated_text.split("\nOut: ")[-1]
                prompt = prompt.split(" Action: ")[0]
                self.vla.frozen_prompt = prompt + " Action: "
                #print('model.frozen_prompt', model.frozen_prompt)
            
            print('openvla_server:', action)
            print('trajectory planning: ', self.vla.frozen_prompt)

            #action, generated_text
            if double_encode:
                response_data = {
                    "action": json_numpy.dumps(action),
                    "trajectory": generated_text  # Replace with your string data
                }
                #return JSONResponse(json_numpy.dumps(action))
                return JSONResponse(response_data)
            else:
                response_data = {
                    "action": action,
                    "trajectory": generated_text  # Replace with your string data
                }
                return JSONResponse(response_data)
            
        except:  # noqa: E722
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'image': np.ndarray, 'instruction': str}\n"
                "You can optionally an `unnorm_key: str` to specific the dataset statistics you want to use for "
                "de-normalizing the output actions."
            )
            return "error"

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.predict_action)
        uvicorn.run(self.app, host=host, port=port)


@dataclass
class DeployConfig:
    # fmt: off
    openvla_path: Union[str, Path] = "/home/cvailab/codebase/embodied-CoT/openvla/openvla-7b"             # HF Hub Path (or path to local run directory)

    # Server Configuration
    host: str = "0.0.0.0"                                               # Host IP Address
    port: int = 8000                                                    # Host Port

    # fmt: on


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = OpenVLAServer(cfg.openvla_path)
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    deploy()
