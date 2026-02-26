#!/bin/bash

export CUDA_VISIBLE_DEVICES=$3

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=$(echo $3 | tr ',' '\n' | wc -l)  # number of GPUs per node

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
llm="Qwen/$1"  # Using HuggingFace model ID

# Training hyperparameters
lr=2e-7
batch_size=4
grad_accum_steps=4

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration (passed as comma-separated list)
datasets=$2

# Calculate dataset length string
num_datasets=$(echo $datasets | tr ',' '\n' | wc -l)
len=$((num_datasets * 10))
len_str="${len}K"

# Output directory
output_dir="Models/$1-VideoJudge-RS-20K-Preference-${len_str}"

# Output configuration
run_name="qwen2vl-baseline-preference-data-ablations"

echo $1
echo "$llm"
echo "$2"
echo "$output_dir"
echo "$NPROC_PER_NODE"
echo "$CUDA_VISIBLE_DEVICES"


# Training arguments
args="--deepspeed ${deepspeed} \
    --model_name_or_path ${llm} \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 2 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --base_interval 1 \
    --video_max_frames 60 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 128000 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --run_name ${run_name} \
    --report_to wandb"

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}
