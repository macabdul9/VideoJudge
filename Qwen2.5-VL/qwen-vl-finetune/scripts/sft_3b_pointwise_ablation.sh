#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=2 #$(nvidia-smi --list-gpus | wc -l)  # Automatically detects available GPUs

DDP_TIMEOUT=60000 # 1000 minutes

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
llm=Qwen/Qwen2.5-VL-3B-Instruct  # Using HuggingFace model ID

# Training hyperparameters
lr=2e-7

batch_size=8
grad_accum_steps=1

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration (replace with public dataset names)
datasets=videojudge0

# Output configuration
run_name="qwen2vl-baseline"
output_dir=Models/Qwen2.5-VL-3B-Instruct-VideoJudge-2K-MaxFrames_$1

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 3 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --base_interval 1 \
    --video_max_frames $1 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 128000 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to wandb"

# Launch training with custom DDP timeout
export TORCH_DISTRIBUTED_TIMEOUT=60000   # 1000 minutes

# Launch training with custom timeout
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --nnodes=${NNODES} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         --rdzv_backend=c10d \
         --max-restarts=30 \
         ${entry_file} ${args}