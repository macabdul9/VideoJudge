from __future__ import annotations

import argparse
import json
from pathlib import Path

from videojudge.data import load_records, write_json


def deduplicate(records: list[dict]) -> list[dict]:
    from datasketch import MinHash, MinHashLSH

    indexes = {}
    kept = []
    for record in records:
        video = record["video_file"]
        if video not in indexes:
            indexes[video] = MinHashLSH(threshold=0.9, num_perm=128)
        signature = MinHash(num_perm=128)
        for token in record["instruction"].split():
            signature.update(token.encode("utf-8"))
        if not indexes[video].query(signature):
            indexes[video].insert(len(kept), signature)
            kept.append(record)
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge and deduplicate the three seed instruction corpora"
    )
    parser.add_argument("--video-instruct", default="MBZUAI/VideoInstruct-100K")
    parser.add_argument("--vcg-plus", default="MBZUAI/VCG-plus_112K")
    parser.add_argument("--videochat2", default="OpenGVLab/VideoChat2-IT")
    parser.add_argument(
        "--descriptions",
        type=Path,
        required=True,
        help="JSON mapping available video filenames to descriptions",
    )
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=25000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file")
    descriptions = json.loads(args.descriptions.read_text())
    records = []
    for row in load_records(args.video_instruct):
        video = next(
            (
                f"{row['video_id']}.{extension}"
                for extension in ["mkv", "mp4", "webm"]
                if (args.video_dir / f"{row['video_id']}.{extension}").is_file()
            ),
            None,
        )
        if video:
            records.append(
                {
                    "id": f"MBZUAI_VideoInstruct-100K_{row['video_id']}",
                    "video_file": video,
                    "instruction": row["q"],
                    "response": row["a"],
                }
            )
    for row in load_records(args.vcg_plus):
        human = [turn["value"] for turn in row["conversations"] if turn["from"] == "human"]
        assistant = [turn["value"] for turn in row["conversations"] if turn["from"] == "gpt"]
        records.append(
            {
                "id": f"MBZUAI_VCG-plus_112K_{row['id']}",
                "video_file": row["video"],
                "instruction": human[-1].split("<video>")[-1].strip(),
                "response": assistant[-1],
            }
        )
    if Path(args.videochat2).is_file():
        chat = load_records(args.videochat2)
    else:
        from datasets import load_dataset

        chat = load_dataset(args.videochat2, "video_conversation", split="videochatgpt")
    for row in chat:
        video = Path(row["video"]).name
        records.append(
            {
                "id": f"OpenGVLab_VideoChat2-IT_{Path(video).stem}",
                "video_file": video,
                "instruction": row["QA"][0]["q"],
                "response": row["QA"][0]["a"],
            }
        )
    records = deduplicate([row for row in records if row["video_file"] in descriptions])
    for index, row in enumerate(records):
        row["id"] += f"__{index}"
        row["video_description"] = descriptions[row["video_file"]]
    from datasets import Dataset

    dataset = Dataset.from_list(records).shuffle(seed=args.seed).shuffle(seed=args.seed)
    if args.num_samples <= 0 or args.num_samples > len(dataset):
        parser.error(f"Requested {args.num_samples} examples; {len(dataset)} are available")
    dataset = dataset.select(range(args.num_samples))
    write_json(args.output, list(dataset))
    print(f"Saved {len(dataset)} seed examples")


if __name__ == "__main__":
    main()
