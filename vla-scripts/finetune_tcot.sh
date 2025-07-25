CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py \
  --vla_path "/home/lixiang/codebase/embodied-CoT/openvla-7b" \
  --data_root_dir ./datasets/libero_new \
  --dataset_name libero_10_end60_retry \
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
  --run_id_note multitask_tcotGL_failuredet_end60_s3w \
  --cofintune_pro 0.3 \
  --method tcot \
  # --use_quantization True


# wandb api: 76c2e4fa1932e31efc4439af88bc25ef2c32c464
#--dataset_name
#[libero_spatial, libero_object, libero_goal, liber_o10, libero, libero_spatial_retry] \

#--cofintune_pro 0.3 0.5

#--method tcot