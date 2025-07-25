CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py \
  --vla_path "/home/lixiang/codebase/embodied-CoT/openvla-7b" \
  --data_root_dir ./datasets/libero_new \
  --dataset_name libero \
  --run_root_dir ./runs \
  --adapter_tmp_dir ./adapter-tmp \
  --lora_rank 32 \
  --batch_size 1  \
  --grad_accumulation_steps 16 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project openvla-libero \
  --wandb_entity lixiang_thu-tsinghua-university \
  --max_steps 30000 \
  --save_steps 5000 \
  --run_id_note openvla-baseline \
  --cofintune_pro 0.3 \
  --method openvla \
  # --use_quantization True

