#!/bin/bash
#SBATCH --job-name=training
#SBATCH -p general
#SBATCH --gres=gpu:A100_80GB:4
#SBATCH --mem=200GB
#SBATCH --time=36:00:00
#SBATCH --output=jobs/%x-%j.out
#SBATCH --error=jobs/%x-%j.err
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --exclude=babel-5-31,shire-1-[1,6,10],shire-2-[5,9]


# -p preempt or -p general
# exclude all shire nodes --exclude=shire
# Optional: Hugging Face cache configuration
export HF_HOME="/data/hf_cache/"
export HF_DATASETS_OFFLINE=0
export HF_DATASETS_CACHE="/data/hf_cache/"
export PIP_CACHE_DIR="/data/hf_cache/"
export NLTK_DATA="/data/hf_cache/"
export TRITON_CACHE_DIR="/data/hf_cache/"
export LD_LIBRARY_PATH=/usr/lib64:/usr/local/cuda-12.5/lib64:$LD_LIBRARY_PATH


source ~/.bashrc
module load cuda-12.5
export LD_LIBRARY_PATH="/usr/lib64:/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Properly source conda.sh to allow conda activate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate venv

# llamafactory-cli train examples/CodeAblation/Qwen3-1.7B-Base-DataMix/code_natural_language_mix_nl_2p5_data_120K.yml
# llamafactory-cli train examples/CodeAblation/Qwen3-1.7B-Base-DataMix/code_natural_language_mix_nl_95_data_120K.yml
# llamafactory-cli train examples/CodeAblation/Qwen3-1.7B-Base-DataMix/code_natural_language_mix_nl_97p5_data_120K.yml
# llamafactory-cli train examples/CodeAblation/Qwen3-1.7B-Base-DataMix/code_natural_language_mix_nl_90_data_120K.yml

# Run the training job with provided YAML config
# llamafactory-cli train $1

python train.py
