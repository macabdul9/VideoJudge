
python run_evaluation.py \
    --model_type qwenvlm \
    --model_name_or_path /data/user_data/abdulw/Qwen2.5-VL-7B-Instruct-VideoJudge-RS-20K \
    --dataset_name_or_path uEval/VideoJudgeVCG-MetaEval \
    --split testmini \
    --video_dir /data/hf_cache/VideoAutoEval/data/VideoChatGPT/videos/ \
    --prompt_fp prompts/evaluation_model.txt \
    --output_dir results_custom

# # # # Running evals with vllm - Qwen2.5-VL-7B-Instruct on VideoJudgeVCG-MetaEval for testmini split
# export CUDA_VISIBLE_DEVICES=0,1
# nohup python run_evaluation.py \
#     --model_type qwenvlm \
#     --model_name_or_path /data/user_data/abdulw/Qwen2.5-VL-7B-Instruct-VideoJudge-RS-20K \
#     --dataset_name_or_path uEval/VideoJudgeVCG-MetaEval \
#     --split testmini \
#     --video_dir /data/hf_cache/VideoAutoEval/data/VideoChatGPT/videos/ \
#     --prompt_fp prompts/evaluation_model.txt \
#     --output_dir results_custom &> logs/qwenvlm_7B_videovcg.log &

# export CUDA_VISIBLE_DEVICES=2,3
# nohup python run_evaluation.py \
#     --model_type qwenvlm \
#     --model_name_or_path /data/user_data/abdulw/Qwen2.5-VL-7B-Instruct-VideoJudge-RS-20K \
#     --dataset_name_or_path uEval/VideoJudgeLLaVA-MetaEval \
#     --split testmini \
#     --video_dir /data/hf_cache/VideoAutoEval/data/LLaVA-Video-Eval/videos/ \
#     --prompt_fp prompts/evaluation_model.txt \
#     --output_dir results_custom &> logs/qwenvlm_7B_llava.log &
