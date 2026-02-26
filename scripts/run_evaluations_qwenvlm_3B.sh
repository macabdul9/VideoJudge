#!/bin/bash
#SBATCH --job-name=videoautoeval
#SBATCH --partition=shire-general
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH --mem=200GB
#SBATCH --time=15:00:00
#SBATCH --output=logs/%A_%a.out


# ## Running evals without vllm - Qwen2.5-VL-32B-Instruct on uEval/VideoJudgeLLaVA-MetaEval for testmini split
# export CUDA_VISIBLE_DEVICES=0
# python run_evaluation.py \
#     --model_type qwenvlm \
#     --model_name_or_path Qwen/Qwen2.5-VL-72B-Instruct \
#     --dataset_name_or_path uEval/VideoJudgeVCG-MetaEval \
#     --split testmini \
#     --video_dir /data/hf_cache/VideoAutoEval/data/VideoChatGPT/videos/ \
#     --prompt_fp prompts/evaluation_model.txt \
#     --output_dir results/Qwen2.5-VL

source ~/anaconda3/etc/profile.d/conda.sh
conda activate videoautoeval

# 2,4,8,16,32,64,128,256 tts_n

# zeroshot 3B
python run_evaluation.py \
    --model_type qwenvlm \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset_name_or_path uEval/VCGLLaVA-1K-Mix \
    --split test \
    --video_dir /data/hf_cache/VideoAutoEval/1KTestVideos \
    --prompt_fp prompts/evaluation_model.txt \
    --output_dir results/Qwen2.5-VL/test_time_scaling \
    --do_sample True \
    --temperature 0.5 \
    --tts_n 16 \
    --use_vllm 


# finetuned 3B
python run_evaluation.py \
    --model_type qwenvlm \
    --model_name_or_path uEval/Qwen2.5-VL-3B-Instruct-VideoJudge-RS-20K \
    --dataset_name_or_path uEval/VCGLLaVA-1K-Mix \
    --split test \
    --video_dir /data/hf_cache/VideoAutoEval/1KTestVideos \
    --prompt_fp prompts/evaluation_model.txt \
    --output_dir results/Qwen2.5-VL/test_time_scaling \
    --do_sample True \
    --temperature 0.5 \
    --tts_n 16 \
    --use_vllm