# export CUDA_VISIBLE_DEVICES=0
# nohup python run_evaluation_preference.py \
#     --model_name_or_path uEval/Qwen2.5-VL-3B-Instruct-VideoJudge-RS-20K-PreferenceWithFeedback \
#     --prompt_fp prompts/train_preference_model_with_feedback.txt &> logs/Qwen2.5-VL-3B-Instruct-VideoJudge-RS-20K-PreferenceWithFeedback.log &

# export CUDA_VISIBLE_DEVICES=0
# python run_evaluation_preference.py \
#     --model_name_or_path uEval/Qwen2.5-VL-3B-Instruct-VideoJudge-RS-20K-Preference \
#     --prompt_fp prompts/train_preference_model.txt


# export CUDA_VISIBLE_DEVICES=0
# python run_evaluation_preference.py \
#     --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
#     --prompt_fp prompts/train_preference_model.txt

# export CUDA_VISIBLE_DEVICES=0
# python run_evaluation_preference.py \
#     --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
#     --prompt_fp prompts/train_preference_model.txt

# export CUDA_VISIBLE_DEVICES=0
# python run_evaluation_preference.py \
#     --model_name_or_path Qwen/Qwen2.5-VL-32B-Instruct \
#     --prompt_fp prompts/train_preference_model.txt

    
#&> logs/Qwen2.5-VL-3B-Instruct-VideoJudge-RS-20K-Preference.log &


# export CUDA_VISIBLE_DEVICES=1
# nohup python run_evaluation_preference.py \
#     --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
#     --prompt_fp prompts/train_preference_model_with_feedback.txt &> logs/Qwen2.5-VL-3B-Instruct.log &

# export CUDA_VISIBLE_DEVICES=0
# nohup python run_evaluation_preference.py \
#     --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
#     --prompt_fp prompts/train_preference_model_with_feedback.txt &> logs/Qwen2.5-VL-7B-Instruct.log &

# export CUDA_VISIBLE_DEVICES=1

# nohup python run_evaluation_preference.py \
#     --model_name_or_path Qwen/Qwen2.5-VL-32B-Instruct \
#     --prompt_fp prompts/train_preference_model_with_feedback.txt &> logs/Qwen2.5-VL-32B-Instruct.log &


# export CUDA_VISIBLE_DEVICES=0
# nohup python run_evaluation_preference.py \
#     --model_name_or_path Qwen/Qwen2.5-VL-72B-Instruct \
#     --prompt_fp prompts/train_preference_model.txt &> logs/Qwen2.5-VL-72B-Instruct_without_feedback.log &


# export CUDA_VISIBLE_DEVICES=0
# python run_evaluation_preference.py \
#     --model_name_or_path uEval2/Qwen2.5-VL-7B-Instruct-VideoJudge-RS-20K-Preference \
#     --prompt_fp prompts/train_preference_model.txt # &> logs/Qwen2.5-VL-7B-Instruct-VideoJudge-RS-20K-Preference_without_feedback.log &


# # To Dareen - We need to run this -> To use all the GPUs available do not set the visible devices
# # # For Babel only --> I have unzipped the videos here : /data/hf_cache/VideoAutoEval/VideoAutoAren
# python run_evaluation_preference.py \
#     --model_name_or_path Qwen/Qwen2.5-VL-72B-Instruct \
#     --dataset_name_or_path uEval/VideoAutoArena-MetaEval-Preference \
#     --prompt_fp prompts/train_preference_model_with_feedback.txt \
#     --video_dir /data/hf_cache/VideoAutoEval/VideoAutoArena



# export CUDA_VISIBLE_DEVICES=0
# nohup python run_evaluation_preference.py \
#     --model_name_or_path uEval/Qwen2.5-VL-3B-Instruct-VideoJudge-RS-20K-PreferenceWithFeedback-200K \
#     --dataset_name_or_path uEval/VideoAutoArena-MetaEval-Preference \
#     --prompt_fp prompts/train_preference_model_with_feedback.txt \
#     --video_dir data/VideoAutoArena/ \
#     --output_dir preference_results/data_ablations/ &> logs/Qwen2.5-VL-3B-Instruct-VideoJudge-RS-20K-PreferenceWithFeedback-200K.log &

# export CUDA_VISIBLE_DEVICES=1
# nohup python run_evaluation_preference.py \
#     --model_name_or_path uEval/Qwen2.5-VL-3B-Instruct-VideoJudge-RS-20K-Preference-200K \
#     --dataset_name_or_path uEval/VideoAutoArena-MetaEval-Preference \
#     --prompt_fp prompts/train_preference_model.txt \
#     --video_dir data/VideoAutoArena/ \
#     --output_dir preference_results/data_ablations/ &> logs/Qwen2.5-VL-3B-Instruct-VideoJudge-RS-20K-Preference-200K.log &




# export CUDA_VISIBLE_DEVICES=1
# nohup python run_evaluation_preference.py \
#     --model_name_or_path uEval2/Qwen2.5-VL-7B-Instruct-VideoJudge-RS-20K-PreferenceWithFeedback-200K \
#     --dataset_name_or_path uEval/VideoAutoArena-MetaEval-Preference \
#     --prompt_fp prompts/train_preference_model_with_feedback.txt \
#     --video_dir data/VideoAutoArena/ \
#     --output_dir preference_results/data_ablations/ &> logs/Qwen2.5-VL-7B-Instruct-VideoJudge-RS-20K-PreferenceWithFeedback-200K.log &


# export CUDA_VISIBLE_DEVICES=1
# nohup python run_evaluation_preference.py \
#     --model_name_or_path uEval2/Qwen2.5-VL-7B-Instruct-VideoJudge-RS-20K-Preference-200K \
#     --dataset_name_or_path uEval/VideoAutoArena-MetaEval-Preference \
#     --prompt_fp prompts/train_preference_model_with_feedback.txt \
#     --video_dir data/VideoAutoArena/ \
#     --output_dir preference_results/data_ablations/ &> logs/Qwen2.5-VL-7B-Instruct-VideoJudge-RS-20K-Preference-200K.log &



export CUDA_VISIBLE_DEVICES=1
nohup python run_evaluation_preference.py \
    --model_name_or_path uEval2/Qwen2.5-VL-7B-Instruct-VideoJudge-RS-20K-Preference-10K \
    --dataset_name_or_path uEval/VideoAutoArena-MetaEval-Preference \
    --prompt_fp prompts/train_preference_model_with_feedback.txt \
    --video_dir data/VideoAutoArena/ \
    --output_dir preference_results/data_ablations/ &> logs/Qwen2.5-VL-7B-Instruct-VideoJudge-RS-20K-Preference-10K.log &
