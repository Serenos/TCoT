import torch
from peft import LoraConfig, PeftModel
from transformers import AutoModelForVision2Seq

adapter_dir = '/home/lixiang/codebase/embodied-CoT/adapter-tmp/openvla-7b+libero_spatial+tcot+libero_spatial_fintune_full_trajv3+cofintune0.5/step_10000'
run_dir = '/home/lixiang/codebase/embodied-CoT/runs/openvla-7b+libero_spatial+tcot+libero_spatial_fintune_full_trajv3+cofintune0.5'
vla_path = '/home/lixiang/codebase/embodied-CoT/openvla-7b'


print(f'merging adapter {adapter_dir} to model {vla_path} ......')
base_vla = AutoModelForVision2Seq.from_pretrained(
    vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
)
merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
merged_vla = merged_vla.merge_and_unload()
merged_vla.save_pretrained(run_dir)