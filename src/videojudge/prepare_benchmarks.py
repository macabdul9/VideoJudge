from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from videojudge.data import load_records, write_json

VCG_CONFIGS = ("Consistency", "Generic", "Temporal")
LLAVA_CONFIGS = (
    "0_30_s_academic_v0_1",
    "0_30_s_youtube_v0_1",
    "0_30_s_activitynet",
    "0_30_s_perceptiontest",
    "0_30_s_nextqa",
    "30_60_s_academic_v0_1",
    "30_60_s_youtube_v0_1",
    "30_60_s_activitynet",
    "30_60_s_perceptiontest",
    "30_60_s_nextqa",
    "1_2_m_youtube_v0_1",
    "1_2_m_academic_v0_1",
    "1_2_m_activitynet",
    "1_2_m_nextqa",
    "2_3_m_youtube_v0_1",
    "2_3_m_academic_v0_1",
    "2_3_m_activitynet",
    "2_3_m_nextqa",
    "llava_hound",
)


def vcg_records(rows: list[dict], config: str, video_dir: Path) -> list[dict]:
    records = []
    for index, row in enumerate(rows):
        video = f"{row['video_name']}.mp4"
        if not (video_dir / video).is_file():
            continue
        if config == "Consistency":
            first = row.get("question_1")
            question = first if first and first != "None" else row["question_2"]
        else:
            question = row["question"]
        records.append(
            {
                "id": f"lmms-lab_VideoChatGPT_{row['video_name']}_{config}__{index}",
                "video_id": video,
                "instruction": question,
                "response": row["answer"],
            }
        )
    return records


def llava_records(rows: list[dict], video_dir: Path) -> list[dict]:
    available = {path.stem for path in video_dir.glob("*.mp4")}
    records = []
    filtered = [row for row in rows if row["id"] in available]
    for index, row in enumerate(filtered):
        human = [turn["value"] for turn in row["conversations"] if turn["from"] == "human"]
        assistant = [turn["value"] for turn in row["conversations"] if turn["from"] == "gpt"]
        instruction = human[-1].replace("<image>", "").replace("<video>", "").strip()
        if not instruction:
            continue
        records.append(
            {
                "id": f"lmms-lab_LLaVA-Video-178K_{row['id']}__{index}",
                "video_id": f"{row['id']}.mp4",
                "instruction": instruction,
                "response": assistant[-1],
            }
        )
    return records


def add_descriptions(records: list[dict], descriptions: dict) -> list[dict]:
    for row in records:
        description = descriptions[row["video_id"]]
        row["video_description"] = (
            description["description"] if isinstance(description, dict) else description
        )
        if isinstance(description, dict) and "model_name" in description:
            row["video_description_model"] = description["model_name"]
    return records


def meta_splits(
    rows: list[dict], seeds: list[dict], descriptions: dict, seed: int = 42, mini_size: int = 5000
) -> dict[str, list[dict]]:
    from datasets import Dataset

    unique = {}
    for row in rows:
        identifier = row["id"] if "id" in row else row["video_id"]
        key = (str(identifier), int(row["gen_rating"]))
        unique.setdefault(key, row)
    counts = Counter(key[0] for key in unique)
    seed_map = {str(row["id"]): row for row in seeds}
    prepared = []
    for (identifier, rating), row in unique.items():
        if counts[identifier] < 5:
            continue
        original = seed_map[identifier]
        video = original.get("video_id", original.get("video_file"))
        if video is None:
            raise ValueError(f"Missing video filename for {identifier}")
        prepared.append(
            {
                **row,
                "gen_rating": rating,
                "instruction": original["instruction"],
                "unique_id": f"{identifier}_gr_{rating}",
                "video_file": video,
                "description": descriptions[video],
            }
        )
    if not prepared:
        raise ValueError("No complete five-rating evaluation groups")
    if not 0 < mini_size <= len(prepared):
        raise ValueError(f"testmini size must be between 1 and {len(prepared)}")
    # Match prepare_eval_data.py: test and testmini each have their own final shuffle.
    once = Dataset.from_list(prepared).shuffle(seed=seed)
    twice = once.shuffle(seed=seed)
    return {
        "test": list(twice),
        "testmini": list(twice.select(range(mini_size)).shuffle(seed=seed)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Construct VideoJudge meta-evaluation datasets")
    commands = parser.add_subparsers(dest="command", required=True)
    seeds = commands.add_parser(
        "seeds", help="Prepare VideoChatGPT or LLaVA-Video seed instructions"
    )
    seeds.add_argument("--benchmark", choices=["vcg", "llava"], required=True)
    seeds.add_argument(
        "--input-dir", type=Path, help="Local CONFIG.json files; otherwise load upstream datasets"
    )
    seeds.add_argument("--configs", nargs="+", help="Explicit upstream dataset configs")
    seeds.add_argument("--video-dir", type=Path, required=True)
    seeds.add_argument("--descriptions", type=Path, required=True)
    seeds.add_argument("--output", type=Path, required=True)
    seeds.add_argument("--seed", type=int, default=42)
    meta = commands.add_parser("meta", help="Convert bootstrapping CSV into test/testmini splits")
    ratings = meta.add_mutually_exclusive_group(required=True)
    ratings.add_argument("--csv", type=Path, help="Original bootstrapping CSV")
    ratings.add_argument("--ratings", type=Path, help="Pointwise JSON from videojudge.finalize")
    meta.add_argument("--seeds", required=True, help="Seed JSON or Hub dataset")
    meta.add_argument("--seed-split", default="test")
    meta.add_argument("--descriptions", type=Path, required=True)
    meta.add_argument("--mini-size", type=int, default=5000)
    meta.add_argument("--seed", type=int, default=42)
    meta.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    descriptions = json.loads(args.descriptions.read_text())
    if args.command == "meta":
        if args.output_dir.exists():
            parser.error("Choose a new output directory")
        if args.csv:
            with args.csv.open(newline="") as source:
                rows = list(csv.DictReader(source))
        else:
            rows = load_records(str(args.ratings))
        splits = meta_splits(
            rows, load_records(args.seeds, args.seed_split), descriptions, args.seed, args.mini_size
        )
        for name, records in splits.items():
            write_json(args.output_dir / f"{name}.json", records)
        print(f"Prepared test={len(splits['test'])}, testmini={len(splits['testmini'])}")
        return
    if args.output.exists():
        parser.error("Choose a new output file")
    configs = args.configs or (VCG_CONFIGS if args.benchmark == "vcg" else LLAVA_CONFIGS)
    all_rows, prepared = [], []
    for config in configs:
        if args.input_dir:
            rows = load_records(str(args.input_dir / f"{config}.json"))
        else:
            from datasets import load_dataset

            dataset = (
                "lmms-lab/VideoChatGPT" if args.benchmark == "vcg" else "lmms-lab/LLaVA-Video-178K"
            )
            split = "test" if args.benchmark == "vcg" else "open_ended"
            rows = list(load_dataset(dataset, config, split=split))
        if args.benchmark == "vcg":
            prepared.extend(vcg_records(rows, config, args.video_dir))
        else:
            all_rows.extend(rows)
    if args.benchmark == "llava":
        prepared = llava_records(all_rows, args.video_dir)
    prepared = add_descriptions(prepared, descriptions)
    if not prepared:
        raise ValueError("No seed instructions matched available videos")
    from datasets import Dataset

    dataset = Dataset.from_list(prepared).shuffle(seed=args.seed).shuffle(seed=args.seed)
    write_json(args.output, list(dataset))
    print(f"Prepared {len(dataset)} {args.benchmark} seed instructions")


if __name__ == "__main__":
    main()
