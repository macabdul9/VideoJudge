from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from videojudge.data import format_prompt, load_records, write_json
from videojudge.parsing import extract_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reasoning for gold seed responses")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    template = Path("prompts/gen_reasoning.txt").read_text()
    if args.output.exists():
        parser.error("Choose a new output file")
    results = {}
    for example in load_records(args.dataset, args.split):
        prompt = format_prompt(
            template,
            instruction=example["instruction"],
            response=example["response"],
            video_description=example["video_description"],
        )
        if args.dry_run:
            print(prompt)
            return
        import requests

        response = requests.post(
            os.environ["ENDPOINT"],
            headers={"Content-Type": "application/json", "api-key": os.environ["API_KEY"]},
            json={
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.6,
                "max_tokens": args.max_tokens,
            },
            timeout=180,
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"]
        thinking = re.search(r"<thinking>.*?</thinking>", text, re.DOTALL)
        score = extract_score(text)
        if thinking is None or score is None:
            raise ValueError(f"Invalid gold reasoning output for {example['id']}")
        results[str(example["id"])] = {"reasoning": thinking[0], "score": score}
        write_json(args.output, results)
    print(json.dumps({"examples": len(results)}))


if __name__ == "__main__":
    main()
