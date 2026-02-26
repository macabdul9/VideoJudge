# export CUDA_VISIBLE_DEVICES=0
# python run_evaluation_preference.py \
#     --model_name_or_path uEval/Qwen2.5-VL-7B-Instruct-VideoJudge-RS-20K-PreferenceWithFeedback \
#     --dataset_name_or_path uEval/VideoRewardBench-Preference \
#     --prompt_fp prompts/train_preference_model_with_feedback.txt \
#     --video_dir data/videorewardbench/ \
#     --output_dir preference_results/videorewardbench/


export CUDA_VISIBLE_DEVICES=1
nohup python run_evaluation_preference.py \
    --model_name_or_path uEval/Qwen2.5-VL-7B-Instruct-VideoJudge-RS-20K-PreferenceWithFeedback \
    --dataset_name_or_path uEval/VideoRewardBench-Preference \
    --prompt_fp prompts/train_preference_model_with_feedback.txt \
    --video_dir data/videorewardbench/ \
    --output_dir preference_results/videorewardbench/ &> logs/Qwen2.5-VL-7B-Instruct-VideoJudge-RS-20K-PreferenceWithFeedback-train_preference_model_with_feedback_videorewardbench.log &


export CUDA_VISIBLE_DEVICES=0
nohup python run_evaluation_preference.py \
    --model_name_or_path Qwen/Qwen2.5-VL-32B-Instruct\
    --dataset_name_or_path uEval/VideoRewardBench-Preference \
    --prompt_fp prompts/train_preference_model_with_feedback.txt \
    --video_dir data/videorewardbench/ \
    --output_dir preference_results/videorewardbench/ &> logs/Qwen2.5-VL-32B-Instruct-train_preference_model_with_feedback_videorewardbench.log &
