export CUDA_VISIBLE_DEVICES=1
# python run_evaluation_unimodal.py \
#     --model_name_or_path Qwen/Qwen3-1.7B \
#     --dataset_name_or_path uEval/VideoJudgeVCG-MetaEval \
#     --output_dir results/Qwen3-1.7B

# python run_evaluation_unimodal.py \
#     --model_name_or_path Qwen/Qwen3-1.7B \
#     --dataset_name_or_path uEval/VideoJudgeLLaVA-MetaEval \
#     --output_dir results/Qwen3-1.7B

# python run_evaluation_unimodal.py \
#     --model_name_or_path Qwen/Qwen3-1.7B \
#     --dataset_name_or_path uEval/VideoJudgeVCG-MetaEval \
#     --output_dir results/Qwen3-1.7B \
#     --enable_thinking \

# python run_evaluation_unimodal.py \
#     --model_name_or_path Qwen/Qwen3-1.7B \
#     --dataset_name_or_path uEval/VideoJudgeLLaVA-MetaEval \
#     --output_dir results/Qwen3-1.7B \
#     --enable_thinking \

export CUDA_VISIBLE_DEVICES=0
python run_evaluation_unimodal.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --dataset_name_or_path uEval/LongVideoBench-MetaEval \
    --output_dir results/Qwen3-1.7B \
    --split test \
    --rating_column is_correct \
    
python run_evaluation_unimodal.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --dataset_name_or_path uEval/LongVideoBench-MetaEval \
    --output_dir results/Qwen3-1.7B \
    --split test \
    --rating_column is_correct \
    --enable_thinking \

export CUDA_VISIBLE_DEVICES=0
nohup python run_evaluation_unimodal.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --dataset_name_or_path uEval/VatexEval-MetaEval \
    --output_dir results/Qwen3-1.7B \
    --split testmini \
    --rating_column score &> logs/Qwen3-1.7B_VatexEval.log &

export CUDA_VISIBLE_DEVICES=3
nohup python run_evaluation_unimodal.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --dataset_name_or_path uEval/VatexEval-MetaEval \
    --output_dir results/Qwen3-1.7B \
    --split testmini \
    --rating_column score \
    --enable_thinking &> logs/Qwen3-1.7B_VatexEval_thinking.log &
