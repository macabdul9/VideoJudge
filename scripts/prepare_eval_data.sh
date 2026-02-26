python prepare_eval_data.py \
    --csv_path outputs/LLaVA-Video-Eval_test_None_0.0_5_use_video_description_True.csv \
    --dataset_name_or_path uEval/LLaVA-Video-Eval-Original \
    --split test \
    --description_file outputs/llavavideo_eval_descriptions.json \
    --final_dataset_name_or_path uEval/VideoJudgeLLaVA-MetaEval