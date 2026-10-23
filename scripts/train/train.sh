#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

mode="${1:?usage: train.sh MODE MODEL DATA_CONFIG OUTPUT_DIR [trainer options ...]}"
model="${2:?Specify a model ID or checkpoint path}"
data_config="${3:?Specify the datasets.json written by data preparation}"
output_dir="${4:?Specify an output directory}"
shift 4
if [[ "$mode" != "pointwise" && "$mode" != "pairwise" ]]; then
    echo "MODE must be pointwise or pairwise" >&2
    exit 1
fi
if [[ ! -f "$data_config" ]]; then
    echo "Data configuration not found: $data_config" >&2
    exit 1
fi
export VIDEOJUDGE_DATA_CONFIG
VIDEOJUDGE_DATA_CONFIG="$(python -c 'import pathlib, sys; print(pathlib.Path(sys.argv[1]).resolve())' "$data_config")"
export PYTHONPATH="$repo_root/training${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

command=(
    torchrun
    --nproc_per_node="${NPROC_PER_NODE:-4}"
    --master_addr="${MASTER_ADDR:-127.0.0.1}"
    --master_port="${MASTER_PORT:-29500}"
    training/qwenvl/train/train_qwen.py
    --model_name_or_path "$model"
    --dataset_use "$mode"
    --output_dir "$output_dir"
    --deepspeed configs/zero3.json
    --data_flatten True
    --tune_mm_vision False
    --tune_mm_mlp True
    --tune_mm_llm True
    --bf16 True
    --num_train_epochs 2
    --per_device_train_batch_size "${MICRO_BATCH_SIZE:-1}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-4}"
    --base_interval 1
    --video_max_frames 60
    --eval_strategy no
    --save_strategy steps
    --save_steps 200
    --save_total_limit 1
    --learning_rate 2e-7
    --weight_decay 0
    --warmup_ratio 0.03
    --max_grad_norm 1
    --lr_scheduler_type cosine
    --logging_steps 1
    --model_max_length 128000
    --gradient_checkpointing True
    --dataloader_num_workers 4
    --report_to none
    "$@"
)
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
mkdir -p "$output_dir"
"${command[@]}" 2>&1 | tee -a "$output_dir/train.log"
