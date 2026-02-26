# export CUDA_VISIBLE_DEVICES=3
# # HuggingFaceTB/SmolVLM2-256M-Video-Instruct,
# models="HuggingFaceTB/SmolVLM2-500M-Video-Instruct,HuggingFaceTB/SmolVLM2-2.2B-Instruct"

# for model in ${models//,/ }; do
#     echo "Running evaluation for model: $model"
#     python evaluate.py \
#         --model_type smolvlm \
#         --model_name_or_path "$model" \
#         --dataset_name_or_path uEval/VideoJudgeVCG-MetaEval \
#         --split testmini \
#         --video_dir /data/hf_cache/VideoAutoEval/data/VideoChatGPT/videos \
#         --prompt_fp prompts/evaluation_model.txt \
#         --output_dir results
# done

export CUDA_VISIBLE_DEVICES=0
python evaluate.py \
    --model_type smolvlm \
    --model_name_or_path "HuggingFaceTB/SmolVLM2-500M-Video-Instruct" \
    --dataset_name_or_path uEval/VideoJudgeVCG-MetaEval \
    --split testmini \
    --video_dir /data/hf_cache/VideoAutoEval/data/VideoChatGPT/videos \
    --prompt_fp prompts/evaluation_model.txt \
    --output_dir results

export CUDA_VISIBLE_DEVICES=1
python evaluate.py \
    --model_type smolvlm \
    --model_name_or_path "HuggingFaceTB/SmolVLM2-2.2B-Instruct" \
    --dataset_name_or_path uEval/VideoJudgeVCG-MetaEval \
    --split testmini \
    --video_dir /data/hf_cache/VideoAutoEval/data/VideoChatGPT/videos \
    --prompt_fp prompts/evaluation_model.txt \
    --output_dir results

## LLava
export CUDA_VISIBLE_DEVICES=3
models="HuggingFaceTB/SmolVLM2-256M-Video-Instruct,HuggingFaceTB/SmolVLM2-500M-Video-Instruct,HuggingFaceTB/SmolVLM2-2.2B-Instruct"
for model in ${models//,/ }; do
    echo "Running evaluation for model: $model"
    python evaluate.py \
        --model_type smolvlm \
        --model_name_or_path "$model" \
        --dataset_name_or_path uEval/VideoJudgeLLaVA-MetaEval \
        --split testmini \
        --video_dir /data/hf_cache/VideoAutoEval/data/LLaVA-Video-Eval/videos/ \
        --prompt_fp prompts/evaluation_model.txt \
        --output_dir results
done
