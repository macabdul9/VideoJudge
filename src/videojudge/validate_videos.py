from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from videojudge.data import write_json


def check_video(path: str) -> dict:
    import av

    try:
        with av.open(path) as container:
            first = next(container.decode(video=0), None)
            if first is None:
                return {"file": path, "error": "No decodable video frame"}
    except (av.error.FFmpegError, OSError, ValueError) as error:
        return {"file": path, "error": str(error)}
    return {"file": path, "error": None}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check that each video has a decodable first frame"
    )
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--exclude-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.exclude_output.exists() or args.report.exists():
        parser.error("Choose new output files")
    paths = sorted(
        str(path)
        for path in args.video_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".mp4", ".mkv", ".webm", ".avi", ".mov"}
    )
    if not paths:
        raise ValueError("No video files")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(check_video, paths))
    invalid = [result for result in results if result["error"] is not None]
    args.exclude_output.parent.mkdir(parents=True, exist_ok=True)
    args.exclude_output.write_text("".join(Path(result["file"]).name + "\n" for result in invalid))
    write_json(args.report, {"checked": len(paths), "invalid": invalid})
    print(json.dumps({"checked": len(paths), "invalid": len(invalid)}))


if __name__ == "__main__":
    main()
