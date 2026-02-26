
# ## Running evals without vllm - Qwen2.5-VL-32B-Instruct on uEval/VideoJudgeLLaVA-MetaEval for testmini split
export CUDA_VISIBLE_DEVICES=0,1,2,3
# python run_evaluation.py \
#     --model_type qwenvlm \
#     --model_name_or_path Qwen/Qwen2.5-VL-72B-Instruct \
#     --dataset_name_or_path uEval/VideoJudgeVCG-MetaEval \
#     --split testmini \
#     --video_dir /data/hf_cache/VideoAutoEval/data/VideoChatGPT/videos/ \
#     --prompt_fp prompts/evaluation_model.txt \
#     --output_dir results/Qwen2.5-VL

python run_evaluation.py \
    --model_type qwenvlm \
    --model_name_or_path Qwen/Qwen2.5-VL-72B-Instruct \
    --dataset_name_or_path uEval/VideoJudgeLLaVA-MetaEval \
    --split testmini \
    --video_dir /data/hf_cache/VideoAutoEval/data/LLaVA-Video-Eval/videos/ \
    --prompt_fp prompts/evaluation_model.txt \
    --output_dir results/Qwen2.5-VL

# export CUDA_VISIBLE_DEVICES=0,1,2,3
# python evaluate.py \
#     --model_type qwenvlm \
#     --model_name_or_path Qwen/Qwen2.5-VL-72B-Instruct \
#     --dataset_name_or_path uEval/VideoJudgeVCG-MetaEval \
#     --split testmini \
#     --video_dir /data/hf_cache/VideoAutoEval/data/VideoChatGPT/videos/ \
#     --prompt_fp prompts/evaluation_model.txt \
#     --output_dir results