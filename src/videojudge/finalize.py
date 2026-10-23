from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

from videojudge.data import write_json

GOLD_REASONING_EXCLUSIONS = (
    "incorrect",
    "incorrectly",
    "not relevant",
    "not appropriate",
    "not satisfactory",
    "not acceptable",
    "not valid",
    "not reliable",
    "not convincing",
    "not persuasive",
    "not coherent",
    "not logical",
    "not consistent",
    "not comprehensive",
    "not thorough",
    "not insightful",
    "not engaging",
    "not interesting",
    "not entertaining",
    "not enjoyable",
    "not clear enough",
    "not well explained",
    "not well articulated",
    "not well structured",
    "not well organized",
    "not well presented",
    "not well supported",
    "not well reasoned",
    "not well justified",
    "not well argued",
    "not well analyzed",
    "not well evaluated",
)


def finalize_examples(bootstrapped: dict, gold_reasoning: dict) -> list[dict]:
    rows = []
    for key, item in bootstrapped.items():
        if set(item["accepted"]) != {"1", "2", "3", "4"}:
            continue
        gold = gold_reasoning.get(key)
        if not gold or float(gold["score"]) != 5:
            continue
        if any(term in gold["reasoning"].lower() for term in GOLD_REASONING_EXCLUSIONS):
            continue
        example = item["example"]
        video = example.get("video_file", example.get("video_id"))
        if video is None:
            raise ValueError(f"Missing video filename for {key}")
        targets = [
            *item["accepted"].values(),
            {
                "response": example["response"],
                "gen_rating": 5,
                "eval_rating": 5,
                "eval_reasoning": gold["reasoning"],
            },
        ]
        for target in targets:
            rows.append(
                {
                    **target,
                    "id": key,
                    "unique_id": f"{key}_gr_{target['gen_rating']}",
                    "video_file": video,
                    "instruction": example["instruction"],
                }
            )
    return rows


def preference_pairs(rows: list[dict], seed: int = 42) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault(row["id"], []).append(row)
    pairs = []
    for key, group in groups.items():
        for first, second in itertools.combinations(group, 2):
            if first["gen_rating"] == second["gen_rating"]:
                continue
            higher, lower = sorted([first, second], key=lambda row: row["gen_rating"], reverse=True)
            pairs.append(
                {
                    "unique_id": f"{key}_{higher['gen_rating']}_{lower['gen_rating']}",
                    "instruction": higher["instruction"],
                    "video_file": higher["video_file"],
                    "selected_response": higher["response"],
                    "rejected_response": lower["response"],
                }
            )
    return random.Random(seed).sample(pairs, len(pairs) // 2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build complete five-rating data and preference pairs"
    )
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument(
        "--gold-reasoning",
        type=Path,
        required=True,
        help="JSON keyed by seed ID with reasoning and score fields",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Choose a new output directory")
    rows = finalize_examples(
        json.loads(args.bootstrap.read_text()), json.loads(args.gold_reasoning.read_text())
    )
    if not rows:
        raise ValueError("No complete accepted examples with score-5 gold reasoning")
    write_json(args.output_dir / "pointwise.json", rows)
    write_json(args.output_dir / "pairwise.json", preference_pairs(rows, args.seed))


if __name__ == "__main__":
    main()
