from __future__ import annotations

import argparse
import json
from pathlib import Path

from videojudge.data import write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate factual video descriptions with Qwen2.5-VL"
    )
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-72B-Instruct")
    parser.add_argument("--fps", type=float, default=2)
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--max-new-tokens", type=int, default=1000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attention", default="flash_attention_2")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    files = sorted(
        path
        for path in args.video_dir.iterdir()
        if path.suffix.lower() in {".mp4", ".mkv", ".webm"}
    )
    if not files:
        raise ValueError("No video files")
    if args.output.exists():
        parser.error("Choose a new output file")
    if min(args.fps, args.max_frames, args.max_new_tokens) <= 0:
        parser.error("Frame and generation settings must be positive")
    system_prompt = Path("prompts/description_system.txt").read_text()
    judge = None
    output = {}
    for path in files:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Using the video content provided, generate a detailed "
                        "and factual video description that strictly reflects the visible content.",
                    },
                    {
                        "type": "video",
                        "video": str(path),
                        "fps": args.fps,
                        "max_frames": args.max_frames,
                        "total_pixels": 20480 * 28 * 28,
                        "min_pixels": 16 * 28 * 28,
                    },
                ],
            },
        ]
        if args.dry_run:
            print(json.dumps(messages, indent=2))
            return
        if judge is None:
            from videojudge.inference import QwenVideoJudge

            judge = QwenVideoJudge(args.model, args.device, attention=args.attention)
        output[path.name] = judge.generate(messages, max_new_tokens=args.max_new_tokens)
        write_json(args.output, output)


if __name__ == "__main__":
    main()
