from __future__ import annotations

import argparse
from pathlib import Path

from videojudge.data import load_records, write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert VATEX-EVAL annotations for pointwise evaluation"
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    videos = {path.stem: path for path in args.video_dir.rglob("*.mp4")}
    rows = []
    missing = 0
    for index, row in enumerate(load_records(str(args.annotations))):
        stem = row["video_id"].lstrip("-")
        path = videos.get(stem)
        if path is None:
            missing += 1
            continue
        rows.append(
            {
                "unique_id": f"{row['video_id']}@{index}",
                "instruction": row["instruction"],
                "response": row["response"],
                "gen_rating": row["score"],
                "video_file": str(path.relative_to(args.video_dir)),
            }
        )
    if args.output.exists():
        parser.error("Choose a new output file")
    if not rows:
        raise ValueError("No annotations match available videos")
    write_json(args.output, rows)
    print(f"Prepared {len(rows)} annotations; {missing} had no matching video")


if __name__ == "__main__":
    main()
