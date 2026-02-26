# # # Running evals with vllm - Qwen2.5-VL-3B-Instruct on VideoJudgeVCG-MetaEval for testmini split
# python evaluate.py \
#     --use_vllm \
#     --model_type qwenvlm \
#     --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
#     --dataset_name_or_path uEval/VideoJudgeVCG-MetaEval \
#     --split testmini \
#     --video_dir data/VideoChatGPT/Test_Videos \
#     --prompt_fp prompts/evaluation_model.txt \
#     --output_dir results \

# ## Running evals without vllm - Qwen2.5-VL-7B-Instruct on VideoJudgeVCG-MetaEval for testmini split
# python evaluate.py \
#     --use_vllm \
#     --model_type qwenvlm \
#     --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
#     --dataset_name_or_path uEval/VideoJudgeVCG-MetaEval \
#     --split testmini \
#     --video_dir data/VideoChatGPT/Test_Videos \
#     --prompt_fp prompts/evaluation_model.txt \
#     --output_dir results \

# ## Running evals without vllm - Qwen2.5-VL-3B-Instruct on uEval/VideoJudgeLLaVA-MetaEval for testmini split
# python evaluate.py \
#     --use_vllm \
#     --model_type qwenvlm \
#     --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
#     --dataset_name_or_path uEval/VideoJudgeLLaVA-MetaEval \
#     --split testmini \
#     --video_dir data/LLaVA-Video-Eval/videos \
#     --prompt_fp prompts/evaluation_model.txt \
#     --output_dir results \

# ## Running evals without vllm - Qwen2.5-VL-7B-Instruct on uEval/VideoJudgeLLaVA-MetaEval for testmini split
# python evaluate.py \
#     --use_vllm \
#     --model_type qwenvlm \
#     --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
#     --dataset_name_or_path uEval/VideoJudgeLLaVA-MetaEval \
#     --split testmini \
#     --video_dir data/LLaVA-Video-Eval/videos \
#     --prompt_fp prompts/evaluation_model.txt \
#     --output_dir results \

# ## Running evals with vllm - Qwen2.5-VL-3B-Instruct on VideoJudgeVCG-MetaEval for testmini split
# export CUDA_VISIBLE_DEVICES=0
# nohup python evaluate.py \
#     --model_type qwenvlm \
#     --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
#     --dataset_name_or_path uEval/VideoJudgeVCG-MetaEval \
#     --split testmini \
#     --video_dir /data/hf_cache/VideoAutoEval/data/VideoChatGPT/videos/ \
#     --prompt_fp prompts/evaluation_model.txt \
#     --output_dir results  &> logs/qwenvlm_videovcg_3b_testmini.log &

# ## Running evals without vllm - Qwen2.5-VL-7B-Instruct on VideoJudgeVCG-MetaEval for testmini split
# export CUDA_VISIBLE_DEVICES=1
# nohup python evaluate.py \
#     --model_type qwenvlm \
#     --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
#     --dataset_name_or_path uEval/VideoJudgeVCG-MetaEval \
#     --split testmini \
#     --video_dir /data/hf_cache/VideoAutoEval/data/VideoChatGPT/videos/ \
#     --prompt_fp prompts/evaluation_model.txt \
#     --output_dir results &> logs/qwenvlm_videovcg_7b_testmini.log &

# ## Running evals without vllm - Qwen2.5-VL-3B-Instruct on uEval/VideoJudgeLLaVA-MetaEval for testmini split
# export CUDA_VISIBLE_DEVICES=2
# nohup python evaluate.py \
#     --model_type qwenvlm \
#     --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
#     --dataset_name_or_path uEval/VideoJudgeLLaVA-MetaEval \
#     --split testmini \
#     --video_dir /data/hf_cache/VideoAutoEval/data/LLaVA-Video-Eval/videos/ \
#     --prompt_fp prompts/evaluation_model.txt \
#     --output_dir results  &> logs/qwenvlm_llava_3b_testmini.log &

# ## Running evals without vllm - Qwen2.5-VL-7B-Instruct on uEval/VideoJudgeLLaVA-MetaEval for testmini split
# export CUDA_VISIBLE_DEVICES=3
# python evaluate.py \
#     --model_type qwenvlm \
#     --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
#     --dataset_name_or_path uEval/VideoJudgeLLaVA-MetaEval \
#     --split testmini \
#     --video_dir /data/hf_cache/VideoAutoEval/data/LLaVA-Video-Eval/videos/ \
#     --prompt_fp prompts/evaluation_model.txt \
#     --output_dir results &> logs/qwenvlm_llava_7b_testmini.log &


# ## Running evals without vllm - Qwen2.5-VL-32B-Instruct on uEval/VideoJudgeLLaVA-MetaEval for testmini split
# export CUDA_VISIBLE_DEVICES=0,1
# nohup python evaluate.py \
#     --model_type qwenvlm \
#     --model_name_or_path Qwen/Qwen2.5-VL-32B-Instruct \
#     --dataset_name_or_path uEval/VideoJudgeVCG-MetaEval \
#     --split testmini \
#     --video_dir /data/hf_cache/VideoAutoEval/data/VideoChatGPT/videos/ \
#     --prompt_fp prompts/evaluation_model.txt \
#     --output_dir results &> logs/qwenvlm_videovcg_32b_testmini.log &

# ## Running evals without vllm - Qwen2.5-VL-32B-Instruct on uEval/VideoJudgeLLaVA-MetaEval for testmini split

export CUDA_VISIBLE_DEVICES=2,3
python evaluate.py \
    --model_type qwenvlm \
    --model_name_or_path Qwen/Qwen2.5-VL-32B-Instruct \
    --dataset_name_or_path uEval/VideoJudgeLLaVA-MetaEval \
    --split testmini \
    --video_dir /data/hf_cache/VideoAutoEval/data/LLaVA-Video-Eval/videos/ \
    --prompt_fp prompts/evaluation_model.txt \
    --output_dir results &> logs/qwenvlm_llava_32b_testmini.log &
