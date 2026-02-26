#!/bin/bash

# 1) “With vLLM” run for up to 10 minutes
echo "Starting WITH vLLM (10 min max)…"
timeout 600s python evaluate.py \
    --use_vllm \
    --model_type qwenvlm \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \

# Check exit status of timeout/command
if [ $? -eq 124 ]; then
  echo "→ WITH vLLM hit the 10 min time limit."
else
  echo "→ WITH vLLM finished before timeout."
fi

# 2) “Without vLLM” run for up to 10 minutes
echo "Starting WITHOUT vLLM (10 min max)…"
timeout 600s python evaluate.py \
    --model_type qwenvlm \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \

if [ $? -eq 124 ]; then
  echo "→ WITHOUT vLLM hit the 10 min time limit."
else
  echo "→ WITHOUT vLLM finished before timeout."
fi

echo "All done."
