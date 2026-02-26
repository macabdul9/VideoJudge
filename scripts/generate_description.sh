# python description_generator.py \
#     --video_path data/VideoGPTVideos \
#     --batch_size 10 \

# nohup python description_generator.py \
#     --video_path data/VideoChatGPT/Test_Videos \
#     --batch_size 1 \
#     --max_frames 100 \
#     --output_file videochatgpt_eval_descriptions.json &> logs/videochatgpt_eval_descriptions.log &

# nohup python description_generator.py \
#     --video_path data/LLaVA-Video-Eval/videos \
#     --batch_size 10 \
#     --max_frames 100 \
#     --output_file llavavideo_eval_descriptions.json &> logs/llavavideo_eval_descriptions.log &


# nohup python description_generator_vllm.py &> logs/vllm_description_generator.log &
# export VLLM_USE_SPAWN=1
# python description_generator_hf.py \
#     --video_path data/LongVideoBench-MetaEval/videos \
#     --output_dir descriptions \
#     --output_file LongVideoBench_Qwen2.5-VL-72B-Instruct_descriptions.json

# python description_generator_hf.py \
#     --video_path data/VideoAutoArena/videos \
#     --output_dir descriptions \
#     --output_file VideoAutoArena_Qwen2.5-VL-72B-Instruct_descriptions.json

export CUDA_VISIBLE_DEVICES=0,1
python description_generator_hf.py \
    --video_path data/VATEX-EVAL/VATEX_EVAL_VIDEOS \
    --output_dir descriptions \
    --output_file VATEX_EVAL_VIDEOS_Qwen2.5-VL-72B-Instruct_descriptions.json