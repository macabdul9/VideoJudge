from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


def sweep_commands(args: argparse.Namespace, grids: dict) -> list[tuple[list[str], dict[str, str]]]:
    key = args.kind.replace("-", "_")
    if key == "preference_subsets":
        key = "preference_subset_labels"
    values = args.values or grids[key]
    commands = []
    for value in values:
        environment = dict(os.environ)
        output = args.output_dir / f"{args.kind}-{value}"
        if args.kind in {"training-frames", "preference-subsets"}:
            mode = "pointwise" if args.kind == "training-frames" else "pairwise"
            config = (
                args.data_config
                if mode == "pointwise"
                else args.data_config_pattern.format(value=value)
            )
            if not config:
                raise ValueError("Training sweeps require their prepared dataset configuration")
            command = [
                "bash",
                f"scripts/train/train_{mode}.sh",
                args.model,
                str(config),
                str(output),
            ]
            if mode == "pointwise":
                # sft_3b_pointwise_ablation.sh uses two GPUs, microbatch 8, and three epochs.
                environment.setdefault("NPROC_PER_NODE", "2")
                environment.setdefault("MICRO_BATCH_SIZE", "8")
                environment.setdefault("GRADIENT_ACCUMULATION_STEPS", "1")
                command.extend(
                    [
                        "--num_train_epochs",
                        "3",
                        "--video_max_frames",
                        str(int(value)),
                        "--per_device_eval_batch_size",
                        "16",
                    ]
                )
            else:
                environment.setdefault("MICRO_BATCH_SIZE", "4")
                environment.setdefault("GRADIENT_ACCUMULATION_STEPS", "4")
                command.extend(
                    [
                        "--save_steps",
                        "500",
                        "--dataloader_num_workers",
                        "16",
                        "--per_device_eval_batch_size",
                        "8",
                    ]
                )
        else:
            if not args.dataset or not args.video_dir:
                raise ValueError("Evaluation sweeps require --dataset and --video-dir")
            frames = int(value) if args.kind == "evaluation-frames" else args.max_frames
            temperature = float(value) if args.kind == "temperature" else args.temperature
            if temperature < 0 or frames <= 0:
                raise ValueError("Temperature must be nonnegative and frame counts positive")
            command = [
                sys.executable,
                "-m",
                "videojudge.evaluate",
                "--mode",
                "pointwise",
                "--model",
                args.model,
                "--dataset",
                args.dataset,
                "--split",
                args.split,
                "--video-dir",
                str(args.video_dir),
                "--output-dir",
                str(output),
                "--prompt",
                str(args.prompt),
                "--fps",
                str(args.fps),
                "--max-frames",
                str(frames),
                "--backend",
                args.backend,
                "--tensor-parallel-size",
                str(args.tensor_parallel_size),
            ]
            if temperature > 0:
                command.extend(["--do-sample", "--temperature", str(temperature)])
        commands.append((command, environment))
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VideoJudge ablation sweeps sequentially")
    parser.add_argument(
        "kind",
        choices=["temperature", "evaluation-frames", "training-frames", "preference-subsets"],
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--split", default="test")
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--data-config", type=Path)
    parser.add_argument(
        "--data-config-pattern",
        default="",
        help="Subset datasets.json path containing {value}, e.g. preference-{value}",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grids", type=Path, default=Path("configs/ablations.json"))
    parser.add_argument("--values", nargs="+", help="Override the configured grid")
    parser.add_argument("--prompt", type=Path, default=Path("prompts/evaluation_model.txt"))
    parser.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--fps", type=float, default=1)
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.kind == "preference-subsets" and "{value}" not in args.data_config_pattern:
        parser.error("--data-config-pattern must contain {value}")
    commands = sweep_commands(args, json.loads(args.grids.read_text()))
    for command, environment in commands:
        print(shlex.join(command), flush=True)
        if args.kind in {"training-frames", "preference-subsets"}:
            settings = {
                name: environment.get(name, "4")
                for name in ["NPROC_PER_NODE", "MICRO_BATCH_SIZE", "GRADIENT_ACCUMULATION_STEPS"]
            }
            print(json.dumps(settings), flush=True)
        if not args.dry_run:
            subprocess.run(command, env=environment, check=True)


if __name__ == "__main__":
    main()
