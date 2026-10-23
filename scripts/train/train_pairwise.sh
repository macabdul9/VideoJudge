#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
model="${1:?usage: train_pairwise.sh MODEL DATA_CONFIG OUTPUT_DIR [trainer options ...]}"
data_config="${2:?Specify a datasets.json file}"
output_dir="${3:?Specify an output directory}"
shift 3
bash "$script_dir/train.sh" pairwise "$model" "$data_config" "$output_dir" "$@"
