# VideoJudge

Code for our **ICLR 2026** paper, [**VideoJudge: Bootstrapping Enables Scalable Supervision of MLLM-as-a-Judge for Video Understanding**](https://openreview.net/pdf?id=31CznLfRIS).

**Abdul Waheed, Zhen Wu, Dareen Safar Alharthi, Seungone Kim, and Bhiksha Raj**

Carnegie Mellon University

[![Paper](https://img.shields.io/badge/Paper-ICLR%202026-blue?logo=readthedocs&logoColor=white)](https://openreview.net/pdf?id=31CznLfRIS)
[![Code](https://img.shields.io/badge/Code-GitHub-181717?logo=github&logoColor=white)](https://github.com/macabdul9/VideoJudge)
[![Datasets](https://img.shields.io/badge/Datasets-Hugging%20Face-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/VideoJudge?tab=datasets)
[![Models](https://img.shields.io/badge/Models-Hugging%20Face-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/VideoJudge?tab=models)

VideoJudge evaluates responses to video-understanding tasks. We bootstrap rating-controlled
responses through a generator–evaluator feedback loop and fine-tune Qwen2.5-VL judges
for pointwise scoring and pairwise preference prediction.

## Installation

Use Python 3.10 or 3.11 for the GPU environments and run the commands below from this repository's root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install -r requirements/inference.txt
```

Install FFmpeg for video decoding. For FlashAttention inference or training, install
`flash-attn==2.7.4.post1` with `--no-build-isolation` after installing PyTorch.
Inference also supports `--attention sdpa`.

For training:

```bash
python -m pip install -r requirements/train.txt
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
```

Use a separate environment for the optional vLLM backend:

```bash
python -m pip install -e .
python -m pip install -r requirements/vllm.txt
```

These dependency versions match the experiment environment where available. GPU
training and inference still need a smoke test on the target CUDA stack.

## Models and datasets

Our models and datasets are hosted in the [VideoJudge organization](https://huggingface.co/VideoJudge).

| Artifact | Dataset or model ID |
| --- | --- |
| Pointwise training data | `VideoJudge/VideoJudge-RS-20K` |
| Pairwise training data | `VideoJudge/VideoJudge-RS-20K-Preference` |
| Rubric training data | `VideoJudge/VideoJudgeWithRubric-RS-20K` |
| VideoJudge-3B with rubric | `VideoJudge/Qwen2.5-VL-3B-Instruct-VideoJudgeWithRubric-RS-20K` |
| VideoJudge-7B with rubric | `VideoJudge/Qwen2.5-VL-7B-Instruct-VideoJudgeWithRubric-RS-20K` |
| VideoJudge-7B pairwise | `VideoJudge/Qwen2.5-VL-7B-Instruct-VideoJudge-RS-20K-Preference-200K` |
| LongVideoBench evaluation data | `VideoJudge/LongVideoBench-MetaEval` |
| VATEX evaluation data | `VideoJudge/VatexEval-MetaEval` |
| Human preference evaluation data | `VideoJudge/VideoJudgeH-PreferenceEval` |

Download and extract the corresponding videos separately. `--video-dir` specifies
the directory against which each example's `video_file` is resolved. Dataset access
may require authentication with your Hugging Face account.

## Evaluate a judge

### Pointwise scoring

```bash
bash scripts/evaluate_pointwise.sh \
    --model VideoJudge/Qwen2.5-VL-3B-Instruct-VideoJudgeWithRubric-RS-20K \
    --dataset VideoJudge/VatexEval-MetaEval \
    --split testmini \
    --video-dir data/vatex/videos \
    --rating-column score \
    --prompt prompts/evaluation_model_with_rubric.txt \
    --output-dir runs/evaluation/pointwise
```

The judge generates reasoning followed by a score in `<score>...</score>`.
Results, raw model responses, and the evaluation configuration are saved under the
output directory. Invalid or missing score tags are recorded as null predictions.

### Pairwise preferences

```bash
bash scripts/evaluate_pairwise.sh \
    --model VideoJudge/Qwen2.5-VL-7B-Instruct-VideoJudge-RS-20K-Preference-200K \
    --dataset VideoJudge/VideoJudgeH-PreferenceEval \
    --video-dir data/preference/videos \
    --output-dir runs/evaluation/pairwise
```

Each example contains `response_a`, `response_b`, and an `answer` label (`A` or `B`).
The judge must return `<answer>A</answer>` or `<answer>B</answer>`.

Both commands accept Hub dataset IDs or local JSON/JSONL files. The default columns
are `unique_id`, `video_file`, and `instruction`, with `response`/`gen_rating` for
pointwise evaluation. Column names can be changed through CLI arguments.

Useful options:

- `--backend vllm --tensor-parallel-size 2`: use two GPUs with vLLM.
- `--fps 1 --max-frames 180`: control frame sampling.
- `--do-sample --temperature 0.5`: enable sampling; decoding is greedy by default.
- `--prompt prompts/evaluation_model_with_rubric.txt`: use rubric-based scoring.
- `--limit 10 --dry-run`: validate paths and inspect a prompt without loading a model.

Re-running an identical command resumes completed example IDs. Choose a fresh output
directory when changing the evaluation configuration.

### VATEX-EVAL

Convert VATEX annotations using the video's filename matching convention:

```bash
python -m videojudge.normalize_vatex \
    --annotations PATH_TO_VATEX_ANNOTATIONS.json \
    --video-dir data/vatex/videos \
    --output data/vatex/evaluation.json

bash scripts/evaluate_pointwise.sh \
    --model PATH_TO_CHECKPOINT \
    --dataset data/vatex/evaluation.json \
    --video-dir data/vatex/videos \
    --output-dir runs/evaluation/vatex
```

The converter reports annotations whose videos are missing. Video and annotation
files remain subject to their original distribution terms.

### LongVideoBench

Use the pointwise evaluator with `--rating-column is_correct`. To compute grouped
correct-versus-distractor metrics, also pass `--group-column COLUMN` where `COLUMN`
identifies the original question shared by its candidate answers.

## Calculate metrics

```bash
bash scripts/summarize.sh \
    --predictions runs/evaluation/pointwise/predictions.json \
    --mode pointwise \
    --output runs/evaluation/pointwise/metrics.json
```

Pointwise metrics include RMSE, MAE, Pearson/Spearman correlations, and ECE. Undefined
correlations are null. ECE follows the experiment script's half-open bin convention,
which excludes predictions at the largest bin edge; it should not be interpreted
as a corrected calibration estimator.

Use `--mode pairwise` for preference accuracy. Invalid model answers count as incorrect
in `accuracy`; `accuracy_valid` reports accuracy among parseable labelled outputs.
Use `--mode longvideo` for the mean score gap and probability of superiority across
complete question groups, with half credit for ties. Every summary reports coverage.

## Prepare training data

Pointwise training examples contain `video_file`, `instruction`, `response`,
`gen_rating`, and `eval_reasoning`:

```bash
bash scripts/prepare_data.sh \
    --dataset VideoJudge/VideoJudge-RS-20K \
    --mode pointwise \
    --video-dir data/TrainingVideos \
    --output-dir data/train/pointwise
```

Pairwise examples contain `video_file`, `instruction`, `selected_response`, and
`rejected_response`:

```bash
bash scripts/prepare_data.sh \
    --dataset VideoJudge/VideoJudge-RS-20K-Preference \
    --mode pairwise \
    --video-dir data/TrainingVideos \
    --output-dir data/train/pairwise
```

The formatter writes the Qwen conversation JSON and a `datasets.json` configuration
for the trainer. Pairwise response order is randomized with a reproducible seed.
Use `--exclude-videos FILE.txt` to exclude known corrupt videos. For rubric
supervision, supply `--rubric-column COLUMN` and the corresponding `--prompt`.

## Train

Training uses the Qwen VL fine-tuning implementation included in `training/qwenvl/`.
The vision encoder is frozen, while the language model and multimodal merger are
trained. Defaults are BF16, two epochs, learning rate 2e-7, cosine decay, warmup 0.03,
maximum context length 128,000, and up to 60 frames sampled at a one-second interval.

Train a pointwise model:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/train_pointwise.sh \
    Qwen/Qwen2.5-VL-3B-Instruct \
    data/train/pointwise/datasets.json \
    runs/train/videojudge-3b-pointwise
```

Train a pairwise model:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/train_pairwise.sh \
    Qwen/Qwen2.5-VL-7B-Instruct \
    data/train/pairwise/datasets.json \
    runs/train/videojudge-7b-pairwise
```

Both launchers default to four GPU processes, microbatch size 1, and gradient
accumulation 4, giving effective batch size 16. Set `NPROC_PER_NODE`,
`MICRO_BATCH_SIZE`, and `GRADIENT_ACCUMULATION_STEPS` for your hardware:

```text
effective batch size = GPU count × microbatch size × gradient accumulation
```

Additional trainer arguments are forwarded after the output directory. For example,
append `--video_max_frames 30` for a frame-count ablation. Use `DRY_RUN=1` to print
the Torchrun command without starting training.

## Bootstrap training responses

Generate factual descriptions for the available training videos:

```bash
python -m videojudge.describe \
    --video-dir data/TrainingVideos \
    --output data/video_descriptions.json
```

This uses Qwen2.5-VL-72B by default; `--model` accepts another checkpoint.
To construct seed instructions from VideoInstruct-100K, VCG-plus, and VideoChat2-IT,
provide that JSON mapping of video filenames to descriptions:

```bash
python -m videojudge.seed \
    --descriptions data/video_descriptions.json \
    --video-dir data/TrainingVideos \
    --num-samples 25000 \
    --output data/seed.json
```

This merges the three corpora and deduplicates instructions within each video using
MinHashLSH (128 permutations, threshold 0.9). Source dataset arguments also accept
local JSON files. VideoChat2 uses its first question–answer turn. Description generation and response bootstrapping are separate steps.

Seed records contain `id`, `instruction`, `response` (the gold answer),
`video_description`, and `video_file` or `video_id` (the video filename).
The description-based pipeline generates responses targeting ratings 1–4, evaluates
an anonymized shuffled set, and refines disagreements. Gold responses supply rating 5.

Configure the Azure-compatible chat completion deployment used for data generation:

```bash
export ENDPOINT='YOUR_DEPLOYMENT_CHAT_COMPLETIONS_URL'
export API_KEY='YOUR_API_KEY'

bash scripts/bootstrap.sh \
    --dataset SEED_DATASET_OR_JSON \
    --threshold 0 \
    --max-iterations 5 \
    --output runs/bootstrap/responses.json
```

The endpoint URL includes the deployment and API version. Requests use the `api-key`
header. The default temperatures are 0.6 for generation and 0 for evaluation.
Add `--dry-run` to inspect the first prompt without making API calls.

Generate reasoning for the gold answers, then keep complete five-rating groups and
sample half of the possible preference pairs. The finalizer applies the original
negative-reasoning term filter to rating-5 responses:

```bash
python -m videojudge.gold_reasoning \
    --dataset SEED_DATASET_OR_JSON \
    --output runs/bootstrap/gold_reasoning.json

python -m videojudge.finalize \
    --bootstrap runs/bootstrap/responses.json \
    --gold-reasoning runs/bootstrap/gold_reasoning.json \
    --output-dir data/bootstrap
```

These API commands are explicit data-generation steps and may incur provider charges.
Bootstrapping checkpoints accepted responses and feedback after each seed example.
Use the released dataset snapshots to reproduce the paper's exact training examples;
new generations can differ from the original API outputs.

## Layout

| Directory | Contents |
| --- | --- |
| `src/videojudge/` | Data preparation, bootstrapping, inference, and metrics |
| `scripts/` | Bash launchers |
| `training/qwenvl/` | Qwen VL fine-tuning implementation |
| `configs/` | DeepSpeed configuration |
| `prompts/` | Plain-text generation, judging, and training prompts |
| `tests/` | CPU checks for parsing, data formatting, refinement, and metrics |

No videos, model weights, raw experiment logs, or bookkeeping manifests are bundled.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream attribution.
Dataset and checkpoint licenses follow their respective sources and base models;
a repository-wide license for this release has not yet been selected.

## Development

```bash
python -m pip install -e '.[dev]'
ruff check src tests
ruff format --check src tests
pytest
```

## Acknowledgements

Our fine-tuning implementation is built on top of [Qwen-VL](https://github.com/QwenLM/Qwen-VL). We thank the Qwen team for releasing their models and training code.

## Citation

```bibtex
@inproceedings{
waheed2026videojudge,
title={VideoJudge: Bootstrapping Enables Scalable Supervision of {MLLM}-as-a-Judge for Video Understanding},
author={Abdul Waheed and Zhen Wu and Dareen Safar Alharthi and Seungone Kim and Bhiksha Raj},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=31CznLfRIS}
}
```
