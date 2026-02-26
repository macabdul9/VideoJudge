# export CUDA_VISIBLE_DEVICES=0
# python run_evaluation_unimodal.py \
#     --model_name_or_path Qwen/Qwen3-0.6B \
#     --dataset_name_or_path uEval/VideoJudgeVCG-MetaEval \
#     --output_dir results/Qwen3-0.6B

# python run_evaluation_unimodal.py \
#     --model_name_or_path Qwen/Qwen3-0.6B \
#     --dataset_name_or_path uEval/VideoJudgeLLaVA-MetaEval \
#     --output_dir results/Qwen3-0.6B

# export CUDA_VISIBLE_DEVICES=3
# python run_evaluation_unimodal.py \
#     --model_name_or_path Qwen/Qwen3-0.6B \
#     --dataset_name_or_path uEval/VideoJudgeVCG-MetaEval \
#     --output_dir results/Qwen3-0.6B \
#     --enable_thinking \


# python run_evaluation_unimodal.py \
#     --model_name_or_path Qwen/Qwen3-0.6B \
#     --dataset_name_or_path uEval/VideoJudgeLLaVA-MetaEval \
#     --output_dir results/Qwen3-0.6B \
#     --enable_thinking \


# export CUDA_VISIBLE_DEVICES=0
# python run_evaluation_unimodal.py \
#     --model_name_or_path Qwen/Qwen3-0.6B \
#     --dataset_name_or_path uEval/LongVideoBench-MetaEval \
#     --output_dir results/Qwen3-0.6B \
#     --split test \
#     --rating_column is_correct \
    
# python run_evaluation_unimodal.py \
#     --model_name_or_path Qwen/Qwen3-0.6B \
#     --dataset_name_or_path uEval/LongVideoBench-MetaEval \
#     --output_dir results/Qwen3-0.6B \
#     --split test \
#     --rating_column is_correct \
#     --enable_thinking \


export CUDA_VISIBLE_DEVICES=2
python run_evaluation_unimodal.py \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --dataset_name_or_path uEval/VatexEval-MetaEval \
    --output_dir results/Qwen3-0.6B \
    --split testmini \
    --rating_column score \

python run_evaluation_unimodal.py \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --dataset_name_or_path uEval/VatexEval-MetaEval \
    --output_dir results/Qwen3-0.6B \
    --split testmini \
    --rating_column score \
    --enable_thinking \