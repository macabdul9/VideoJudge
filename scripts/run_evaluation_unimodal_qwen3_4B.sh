# export CUDA_VISIBLE_DEVICES=2
# python run_evaluation_unimodal.py \
#     --model_name_or_path Qwen/Qwen3-4B \
#     --dataset_name_or_path uEval/VideoJudgeVCG-MetaEval \
#     --output_dir results/Qwen3-4B

# python run_evaluation_unimodal.py \
#     --model_name_or_path Qwen/Qwen3-4B \
#     --dataset_name_or_path uEval/VideoJudgeLLaVA-MetaEval \
#     --output_dir results/Qwen3-4B

# python run_evaluation_unimodal.py \
#     --model_name_or_path Qwen/Qwen3-4B \
#     --dataset_name_or_path uEval/VideoJudgeVCG-MetaEval \
#     --output_dir results/Qwen3-4B \
#     --enable_thinking \

export CUDA_VISIBLE_DEVICES=2
python run_evaluation_unimodal.py \
    --model_name_or_path Qwen/Qwen3-4B \
    --dataset_name_or_path uEval/VideoJudgeLLaVA-MetaEval \
    --output_dir results/Qwen3-4B \
    --enable_thinking \

# python run_evaluation_unimodal.py \
#     --model_name_or_path Qwen/Qwen3-4B \
#     --dataset_name_or_path uEval/LongVideoBench-MetaEval \
#     --output_dir results/Qwen3-4B \
#     --split test \
#     --rating_column is_correct \
    
# export CUDA_VISIBLE_DEVICES=2
# nohup python run_evaluation_unimodal.py \
#     --model_name_or_path Qwen/Qwen3-4B \
#     --dataset_name_or_path uEval/LongVideoBench-MetaEval \
#     --output_dir results/Qwen3-4B \
#     --split test \
#     --rating_column is_correct \
#     --enable_thinking &> logs/Qwen3-4B_LongVideoBench.log &


# export CUDA_VISIBLE_DEVICES=3
# nohup python run_evaluation_unimodal.py \
#     --model_name_or_path Qwen/Qwen3-4B \
#     --dataset_name_or_path uEval/VatexEval-MetaEval \
#     --output_dir results/Qwen3-4B \
#     --split testmini \
#     --rating_column score &> logs/Qwen3-4B_VatexEval.log &

# export CUDA_VISIBLE_DEVICES=0
# nohup python run_evaluation_unimodal.py \
#     --model_name_or_path Qwen/Qwen3-4B \
#     --dataset_name_or_path uEval/VatexEval-MetaEval \
#     --output_dir results/Qwen3-4B \
#     --split testmini \
#     --rating_column score \
#     --enable_thinking &> logs/Qwen3-4B_VatexEval.log &


# python run_evaluation_unimodal.py \
#     --model_name_or_path Qwen/Qwen3-4B \
#     --dataset_name_or_path uEval/LongVideoBench-MetaEval \
#     --output_dir results/Qwen3-4B \
#     --split test \
#     --rating_column is_correct \
