from __future__ import annotations

import argparse
import random
from pathlib import Path

from videojudge.data import format_prompt, load_records, write_json


def training_record(
    example: dict,
    mode: str,
    video_dir: Path,
    template: str,
    rng: random.Random,
    reasoning_column: str = "eval_reasoning",
    rubric_column: str | None = None,
    pairwise_reasoning_column: str | None = None,
) -> dict:
    video = video_dir / example["video_file"]
    if not video.is_file():
        raise FileNotFoundError(video)
    values = {"instruction": example["instruction"]}
    if mode == "pointwise":
        values["response"] = example["response"]
        rating = float(example["gen_rating"])
        if not rating.is_integer() or not 1 <= rating <= 5:
            raise ValueError("Training ratings must be integers from 1 to 5")
        reasoning = example[reasoning_column]
        target = f"{reasoning}\n<score>{int(rating)}</score>"
        if rubric_column:
            target = f"<rubric>{example[rubric_column]}</rubric>\n{target}"
    elif pairwise_reasoning_column:
        answer = str(example["answer"]).strip().upper()
        if answer not in {"A", "B"}:
            raise ValueError("Pairwise answers must be A or B")
        values.update(response_a=example["response_a"], response_b=example["response_b"])
        reasoning = example[pairwise_reasoning_column].strip()
        if not reasoning:
            raise ValueError("Pairwise reasoning must not be empty")
        if not reasoning.startswith("<thinking>"):
            reasoning = f"<thinking>{reasoning}</thinking>"
        target = f"{reasoning}\n<answer>{answer}</answer>"
    else:
        selected, rejected = example["selected_response"], example["rejected_response"]
        if rng.random() < 0.5:
            values.update(response_a=selected, response_b=rejected)
            answer = "A"
        else:
            values.update(response_a=rejected, response_b=selected)
            answer = "B"
        target = f"<answer>{answer}</answer>"
    prompt = format_prompt(template.removeprefix("<video>\n"), **values)
    return {
        "video": str(video.resolve()),
        "conversations": [
            {"from": "human", "value": "<video>\n" + prompt},
            {"from": "gpt", "value": target},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Qwen VideoJudge fine-tuning data")
    parser.add_argument("--dataset", required=True, help="Hub ID or local JSON/JSONL")
    parser.add_argument("--revision")
    parser.add_argument("--split", default="train")
    parser.add_argument("--mode", choices=["pointwise", "pairwise"], default="pointwise")
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", type=Path)
    parser.add_argument("--reasoning-column", default="eval_reasoning")
    parser.add_argument("--rubric-column")
    parser.add_argument(
        "--pairwise-reasoning-column",
        help="Use fixed response_a/response_b/answer pairs with the supplied reasoning column",
    )
    parser.add_argument(
        "--exclude-videos", type=Path, help="Text file of video filenames to exclude"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Use a prefix after the mode's existing shuffle/filter steps",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Choose a new output directory to avoid overwriting training data")
    if args.pairwise_reasoning_column and args.mode != "pairwise":
        parser.error("--pairwise-reasoning-column requires --mode pairwise")
    prompt = args.prompt or Path(
        "prompts/evaluation_model.txt"
        if args.mode == "pointwise"
        else (
            "prompts/train_preference_model_with_feedback.txt"
            if args.pairwise_reasoning_column
            else "prompts/train_preference_model.txt"
        )
    )
    if args.rubric_column and args.prompt is None:
        parser.error("Supply the rubric-training prompt with --prompt")
    excluded = (
        {
            Path(line.strip()).name
            for line in args.exclude_videos.read_text().splitlines()
            if line.strip()
        }
        if args.exclude_videos
        else set()
    )
    records = load_records(args.dataset, args.split, args.revision)
    if args.mode == "pairwise":
        from datasets import Dataset

        records = list(Dataset.from_list(records).shuffle(seed=args.seed))
    records = [row for row in records if Path(row["video_file"]).name not in excluded]
    if args.max_samples is not None:
        if not 0 < args.max_samples <= len(records):
            parser.error(f"--max-samples must be between 1 and {len(records)}")
        records = records[: args.max_samples]
    rng = random.Random(args.seed)
    template = prompt.read_text()
    formatted = [
        training_record(
            row,
            args.mode,
            args.video_dir,
            template,
            rng,
            args.reasoning_column,
            args.rubric_column,
            args.pairwise_reasoning_column,
        )
        for row in records
    ]
    if not formatted:
        raise ValueError("No training examples after filtering")
    annotation = args.output_dir / f"{args.mode}.json"
    write_json(annotation, formatted)
    write_json(
        args.output_dir / "datasets.json",
        {args.mode: {"annotation_path": str(annotation.resolve()), "data_path": ""}},
    )
    print(f"Prepared {len(formatted)} examples in {annotation}")


if __name__ == "__main__":
    main()
