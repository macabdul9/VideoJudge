from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


def benchmark_commands(
    model: str,
    selected: dict,
    output_dir: Path,
    backend: str,
    prompt: Path | None,
    tensor_parallel_size: int = 1,
) -> list[list[str]]:
    commands = []
    for name, config in selected.items():
        command = [
            sys.executable,
            "-m",
            "videojudge.evaluate",
            "--model",
            model,
            "--dataset",
            config["dataset"],
            "--video-dir",
            config["video_dir"],
            "--mode",
            config["mode"],
            "--split",
            config["split"],
            "--output-dir",
            str(output_dir / name),
            "--backend",
            backend,
            "--tensor-parallel-size",
            str(tensor_parallel_size),
        ]
        for key in ["rating_column", "group_column", "id_column"]:
            if key in config:
                command.extend(["--" + key.replace("_", "-"), config[key]])
        if prompt:
            command.extend(["--prompt", str(prompt)])
        commands.append(command)
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a configured benchmark suite for a judge")
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/benchmarks.json"))
    parser.add_argument("--benchmarks", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", type=Path)
    parser.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    configs = json.loads(args.config.read_text())
    selected = {name: configs[name] for name in args.benchmarks}
    if len({config["mode"] for config in selected.values()}) != 1:
        parser.error("Use separate runs and checkpoints for pointwise and pairwise benchmarks")
    for command in benchmark_commands(
        args.model, selected, args.output_dir, args.backend, args.prompt, args.tensor_parallel_size
    ):
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
