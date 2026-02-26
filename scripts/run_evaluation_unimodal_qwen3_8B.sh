# export CUDA_VISIBLE_DEVICES=3
# python run_evaluation_unimodal.py \
#     --model_name_or_path Qwen/Qwen3-8B \
#     --dataset_name_or_path uEval/VideoJudgeVCG-MetaEval \
#     --output_dir results/Qwen3-8B

# python run_evaluation_unimodal.py \
#     --model_name_or_path Qwen/Qwen3-8B \
#     --dataset_name_or_path uEval/VideoJudgeLLaVA-MetaEval \
#     --output_dir results/Qwen3-8B

# export CUDA_VISIBLE_DEVICES=3
# python run_evaluation_unimodal.py \
#     --model_name_or_path Qwen/Qwen3-8B \
#     --dataset_name_or_path uEval/VideoJudgeVCG-MetaEval \
#     --output_dir results/Qwen3-8B \
#     --enable_thinking \

# python run_evaluation_unimodal.py \
#     --model_name_or_path Qwen/Qwen3-8B \
#     --dataset_name_or_path uEval/VideoJudgeLLaVA-MetaEval \
#     --output_dir results/Qwen3-8B \
#     --enable_thinking \


export CUDA_VISIBLE_DEVICES=3
python run_evaluation_unimodal.py \
    --model_name_or_path Qwen/Qwen3-8B \
    --dataset_name_or_path uEval/LongVideoBench-MetaEval \
    --output_dir results/Qwen3-8B \
    --split test \
    --rating_column is_correct \
    

# python run_evaluation_unimodal.py \
#     --model_name_or_path Qwen/Qwen3-8B \
#     --dataset_name_or_path uEval/LongVideoBench-MetaEval \
#     --output_dir results/Qwen3-8B \
#     --split test \
#     --rating_column is_correct \
#     --enable_thinking \

python run_evaluation_unimodal.py \
    --model_name_or_path Qwen/Qwen3-8B \
    --dataset_name_or_path uEval/VatexEval-MetaEval \
    --output_dir results/Qwen3-8B \
    --split testmini \
    --rating_column score \