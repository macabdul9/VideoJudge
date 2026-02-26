
# debug
# export CUDA_VISIBLE_DEVICES=0
# python run_evaluation.py \
#     --model_type qwenvlm \
#     --model_name_or_path uEval/Qwen2.5-VL-7B-Instruct-VideoJudge-RS-20K \
#     --dataset_name_or_path uEval/VideoAutoArena-MetaEval \
#     --split test \
#     --instruction_column instruction \
#     --response_column response \
#     --rating_column selected \
#     --video_column video_file \
#     --id_column unique_id \
#     --video_dir data/VideoAutoArena/videos/ \
#     --prompt_fp prompts/evaluation_model.txt \
#     --output_dir results_custom

export CUDA_VISIBLE_DEVICES=0
nohup python run_evaluation.py \
    --model_type qwenvlm \
    --model_name_or_path uEval/Qwen2.5-VL-7B-Instruct-VideoJudge-RS-20K \
    --dataset_name_or_path uEval/VideoAutoArena-MetaEval \
    --split test \
    --instruction_column instruction \
    --response_column response \
    --rating_column selected \
    --video_column video_file \
    --id_column unique_id \
    --video_dir data/VideoAutoArena/videos/ \
    --prompt_fp prompts/evaluation_model.txt \
    --output_dir results_custom &> logs/Qwen2.5-VL-7B-Instruct-VideoJudge-RS-20K_VideoAutoArena-MetaEval.log &

export CUDA_VISIBLE_DEVICES=1
python run_evaluation.py \
    --model_type qwenvlm \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset_name_or_path uEval/VideoAutoArena-MetaEval \
    --split test \
    --instruction_column instruction \
    --response_column response \
    --rating_column selected \
    --video_column video_file \
    --id_column unique_id \
    --video_dir data/VideoAutoArena/videos/ \
    --prompt_fp prompts/evaluation_model.txt \
    --output_dir results_custom &> logs/Qwen2.5-VL-7B-Instruct_VideoAutoArena-MetaEval.log &

export CUDA_VISIBLE_DEVICES=2
python run_evaluation.py \
    --model_type qwenvlm \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset_name_or_path uEval/VideoAutoArena-MetaEval \
    --split test \
    --instruction_column instruction \
    --response_column response \
    --rating_column selected \
    --video_column video_file \
    --id_column unique_id \
    --video_dir data/VideoAutoArena/videos/ \
    --prompt_fp prompts/evaluation_model.txt \
    --output_dir results_custom &> logs/Qwen2.5-VL-3B-Instruct_VideoAutoArena-MetaEval.log &

export CUDA_VISIBLE_DEVICES=3
python run_evaluation.py \
    --model_type qwenvlm \
    --model_name_or_path uEval/Qwen2.5-VL-3B-Instruct-VideoJudge-RS-20K \
    --dataset_name_or_path uEval/VideoAutoArena-MetaEval \
    --split test \
    --instruction_column instruction \
    --response_column response \
    --rating_column selected \
    --video_column video_file \
    --id_column unique_id \
    --video_dir data/VideoAutoArena/videos/ \
    --prompt_fp prompts/evaluation_model.txt \
    --output_dir results_custom &> logs/Qwen2.5-VL-3B-Instruct-VideoJudge-RS-20K_VideoAutoArena-MetaEval.log &

# export CUDA_VISIBLE_DEVICES=0,1
# nohup python run_evaluation.py \
#     --model_type qwenvlm \
#     --model_name_or_path Qwen/Qwen2.5-VL-32B-Instruct \
#     --dataset_name_or_path uEval/VideoAutoArena-MetaEval \
#     --split test \
#     --instruction_column instruction \
#     --response_column response \
#     --rating_column selected \
#     --video_column video_file \
#     --id_column unique_id \
#     --video_dir data/VideoAutoArena/videos/ \
#     --prompt_fp prompts/evaluation_model.txt \
#     --output_dir results_custom  &> logs/VideoAutoArena-MetaEval_Qwen2.5-VL-32B-Instruct.log &

# export CUDA_VISIBLE_DEVICES=2,3
# nohup python run_evaluation.py \
#     --model_type qwenvlm \
#     --model_name_or_path Qwen/Qwen2.5-VL-72B-Instruct \
#     --dataset_name_or_path uEval/VideoAutoArena-MetaEval \
#     --split test \
#     --instruction_column instruction \
#     --response_column response \
#     --rating_column selected \
#     --video_column video_file \
#     --id_column unique_id \
#     --video_dir data/VideoAutoArena/videos/ \
#     --prompt_fp prompts/evaluation_model.txt \
#     --output_dir results_custom  &> logs/VideoAutoArena-MetaEval_Qwen2.5-VL-72B-Instruct.log &

