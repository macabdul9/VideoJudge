# python bootstrapping.py \
#     --dataset_name_or_path uEval/VideoInstruct-100K-VCGPlus-VideoChat2-IT-RS-25K \
#     --video_description_column video_description \
#     --use_video_description \
#     --gen_prompt_fp prompts/gen_prompt_desc.txt \
#     --eval_prompt_fp prompts/eval_prompt_desc.txt \
#     --feedback_gen_prompt_fp prompts/feedback_gen_prompt_desc.txt


# Bootstrapping for randomly sampled data with video descriptions
# nohup python bootstrapping.py \
#     --dataset_name_or_path uEval/VideoInstruct-100K-VCGPlus-VideoChat2-IT-RS-25K \
#     --video_description_column video_description \
#     --use_video_description \
#     --gen_prompt_fp prompts/gen_prompt_desc.txt \
#     --eval_prompt_fp prompts/eval_prompt_desc.txt \
#     --feedback_gen_prompt_fp prompts/feedback_gen_prompt_desc.txt &> logs/bootstrap_video_description_train.log &


# nohup python bootstrapping.py \
#     --dataset_name_or_path uEval/VideoChatGPT-Eval \
#     --split test \
#     --video_description_column video_description \
#     --use_video_description \
#     --gen_prompt_fp prompts/gen_prompt_desc.txt \
#     --eval_prompt_fp prompts/eval_prompt_desc.txt \
#     --feedback_gen_prompt_fp prompts/feedback_gen_prompt_desc.txt &> logs/bootstrap_video_description_videochatgpt.log &

# nohup python bootstrapping.py \
#     --dataset_name_or_path uEval/LLaVA-Video-Eval \
#     --split test \
#     --video_description_column video_description \
#     --use_video_description \
#     --gen_prompt_fp prompts/gen_prompt_desc.txt \
#     --eval_prompt_fp prompts/eval_prompt_desc.txt \
#     --feedback_gen_prompt_fp prompts/feedback_gen_prompt_desc.txt &> logs/bootstrap_video_description_llava.log &


nohup python boostrapping_pl.py \
    --dataset_name_or_path uEval/VideoInstruct-100K-VCGPlus-VideoChat2-IT-RS-25K \
    --video_description_column video_description \
    --use_video_description \
    --gen_prompt_fp prompts/gen_prompt_desc.txt \
    --eval_prompt_fp prompts/eval_prompt_desc.txt \
    --feedback_gen_prompt_fp prompts/feedback_gen_prompt_desc.txt &> logs/bootstrap_video_description_train_final.log &