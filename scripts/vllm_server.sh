vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
    --port 8000 \
    --host 0.0.0.0 \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --max_model_len 32768 \
    --limit-mm-per-prompt image=250,video=10