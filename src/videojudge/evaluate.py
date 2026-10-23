from __future__ import annotations

import argparse
import json
from pathlib import Path

from videojudge.data import format_prompt, load_records, write_json
from videojudge.parsing import extract_answer, extract_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Pointwise or pairwise video judge evaluation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True, help="Hub ID or local JSON/JSONL")
    parser.add_argument("--revision")
    parser.add_argument("--split", default="test")
    parser.add_argument("--mode", choices=["pointwise", "pairwise"], default="pointwise")
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", type=Path)
    parser.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--attention", default="flash_attention_2", choices=["sdpa", "flash_attention_2"]
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--fps", type=float, default=1)
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--total-pixels", type=int, default=20480 * 28 * 28)
    parser.add_argument("--min-pixels", type=int, default=16 * 28 * 28)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--id-column", default="unique_id")
    parser.add_argument("--video-column", default="video_file")
    parser.add_argument("--instruction-column", default="instruction")
    parser.add_argument("--response-column", default="response")
    parser.add_argument("--rating-column", default="gen_rating")
    parser.add_argument("--response-a-column", default="response_a")
    parser.add_argument("--response-b-column", default="response_b")
    parser.add_argument("--answer-column", default="answer")
    parser.add_argument("--group-column", help="Question ID for LongVideoBench aggregation")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate data and print the first prompt"
    )
    args = parser.parse_args()
    if min(args.fps, args.max_frames, args.checkpoint_every, args.max_new_tokens) <= 0:
        parser.error("Frame, generation, and checkpoint settings must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.do_sample and args.temperature <= 0:
        parser.error("Sampling requires a positive temperature")
    prompt_path = args.prompt or Path(
        "prompts/evaluation_model.txt"
        if args.mode == "pointwise"
        else "prompts/train_preference_model.txt"
    )
    template = prompt_path.read_text().removeprefix("<video>\n")
    records = load_records(args.dataset, args.split, args.revision)
    if args.limit:
        records = records[: args.limit]
    prepared = []
    seen = set()
    for record in records:
        key = str(record[args.id_column])
        if key in seen:
            raise ValueError(f"Duplicate example ID: {key}")
        seen.add(key)
        values = {"instruction": record[args.instruction_column]}
        if args.mode == "pointwise":
            values["response"] = record[args.response_column]
        else:
            values.update(
                response_a=record[args.response_a_column], response_b=record[args.response_b_column]
            )
        video = args.video_dir / record[args.video_column]
        if not video.is_file():
            raise FileNotFoundError(video)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": format_prompt(template, **values)},
                    {
                        "type": "video",
                        "video": str(video),
                        "fps": args.fps,
                        "max_frames": args.max_frames,
                        "total_pixels": args.total_pixels,
                        "min_pixels": args.min_pixels,
                    },
                ],
            }
        ]
        prepared.append((key, record, messages))
    if not prepared:
        raise ValueError("No evaluation examples")
    if args.dry_run:
        print(json.dumps(prepared[0][2], indent=2))
        return
    settings = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    settings["prompt_text"] = template
    config_file = args.output_dir / "config.json"
    result_file = args.output_dir / "predictions.json"
    if config_file.exists() and json.loads(config_file.read_text()) != settings:
        raise ValueError("Output directory belongs to another evaluation configuration")
    if result_file.exists() and not config_file.exists():
        raise ValueError("Predictions exist without a configuration; choose a new output directory")
    output = json.loads(result_file.read_text()) if result_file.exists() else {}
    write_json(config_file, settings)
    from videojudge.inference import QwenVideoJudge

    judge = QwenVideoJudge(
        args.model, args.device, args.backend, args.tensor_parallel_size, args.attention
    )
    from tqdm import tqdm

    for index, (key, record, messages) in enumerate(tqdm(prepared), 1):
        if key in output:
            continue
        response = judge.generate(messages, args.max_new_tokens, args.temperature, args.do_sample)
        prediction = (
            extract_score(response) if args.mode == "pointwise" else extract_answer(response)
        )
        result = {
            "instruction": record[args.instruction_column],
            "video_file": record[args.video_column],
            "predicted_rating": prediction,
            "eval_output": response,
        }
        if args.mode == "pointwise":
            result.update(
                response=record[args.response_column], true_rating=record.get(args.rating_column)
            )
        else:
            result.update(
                response_a=record[args.response_a_column],
                response_b=record[args.response_b_column],
                true_answer=record.get(args.answer_column),
            )
        if args.group_column:
            result["group_id"] = record[args.group_column]
        output[key] = result
        if index % args.checkpoint_every == 0:
            write_json(result_file, output)
    write_json(result_file, output)
    print(f"Saved predictions to {result_file}")


if __name__ == "__main__":
    main()
