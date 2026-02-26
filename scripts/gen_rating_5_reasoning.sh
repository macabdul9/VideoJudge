# batch size is concurrency level -- this is the number of examples processed in parallel
python generate_reasoning.py \
    --dataset_name_or_path uEval/VideoInstruct-100K-VCGPlus-VideoChat2-IT-RS-25K-Original \
    --split train \
    --batch_size 500 \