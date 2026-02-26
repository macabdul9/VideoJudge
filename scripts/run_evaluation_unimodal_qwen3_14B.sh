# export CUDA_VISIBLE_DEVICES=0
# python run_evaluation_unimodal.py \
#     --model_name_or_path Qwen/Qwen3-14B \
#     --dataset_name_or_path uEval/VideoJudgeVCG-MetaEval \
#     --output_dir results/Qwen3-14B

# python run_evaluation_unimodal.py \
#     --model_name_or_path Qwen/Qwen3-14B \
#     --dataset_name_or_path uEval/VideoJudgeLLaVA-MetaEval \
#     --output_dir results/Qwen3-14B

# export CUDA_VISIBLE_DEVICES=0
# python run_evaluation_unimodal.py \
#     --model_name_or_path Qwen/Qwen3-14B \
#     --dataset_name_or_path uEval/VideoJudgeVCG-MetaEval \
#     --output_dir results/Qwen3-14B \
#     --enable_thinking \

# python run_evaluation_unimodal.py \
#     --model_name_or_path Qwen/Qwen3-14B \
#     --dataset_name_or_path uEval/VideoJudgeLLaVA-MetaEval \
#     --output_dir results/Qwen3-14B \
#     --enable_thinking \

# python run_evaluation_unimodal.py \
#     --model_name_or_path Qwen/Qwen3-14B \
#     --dataset_name_or_path uEval/LongVideoBench-MetaEval \
#     --output_dir results/Qwen3-4B \
#     --split test \
#     --rating_column is_correct


export CUDA_VISIBLE_DEVICES=0
nohup python run_evaluation_unimodal.py \
    --model_name_or_path Qwen/Qwen3-14B \
    --dataset_name_or_path uEval/VatexEval-MetaEval \
    --output_dir results/Qwen3-14B \
    --split testmini \
    --rating_column score &> logs/Qwen3-14B_VatexEval.log &


export CUDA_VISIBLE_DEVICES=1
python run_evaluation_unimodal.py \
    --model_name_or_path Qwen/Qwen3-14B \
    --dataset_name_or_path uEval/LongVideoBench-MetaEval \
    --output_dir results/Qwen3-14B \
    --split test \
    --rating_column is_correct &> logs/Qwen3-14B_LongVideoBench.log &