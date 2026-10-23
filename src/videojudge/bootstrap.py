from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from videojudge.data import format_prompt, load_records, write_json


def accepted_rating(intended: int, evaluated: int, threshold: float) -> bool:
    return 1 <= evaluated <= 5 and abs(intended - evaluated) <= threshold


def generate_json(messages: list[dict], temperature: float, max_tokens: int) -> dict:
    import requests
    from json_repair import repair_json

    endpoint = os.environ["ENDPOINT"]
    api_key = os.environ["API_KEY"]
    response = requests.post(
        endpoint,
        headers={"Content-Type": "application/json", "api-key": api_key},
        json={"messages": messages, "temperature": temperature, "max_tokens": max_tokens},
        timeout=180,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return json.loads(repair_json(content))


def bootstrap_example(
    example: dict,
    rng: random.Random,
    threshold: float,
    max_iterations: int,
    generation_template: str,
    evaluation_template: str,
    feedback_template: str,
    max_tokens: int,
    generation_temperature: float = 0.6,
    evaluation_temperature: float = 0.0,
) -> dict:
    values = {
        "instruction": example["instruction"],
        "gold_standard_response": example["response"],
        "video_description": example["video_description"],
    }
    messages = [{"role": "user", "content": format_prompt(generation_template, **values)}]
    accepted = {}
    history = []
    pending = {f"rating_{rating}" for rating in range(1, 5)}
    for iteration in range(max_iterations + 1):
        candidates = generate_json(messages, generation_temperature, max_tokens)
        if not isinstance(candidates, dict) or not candidates:
            raise ValueError("Generator must return a nonempty rating-to-response object")
        if set(candidates) != pending:
            raise ValueError(f"Expected generated response keys: {sorted(pending)}")
        messages.append({"role": "assistant", "content": json.dumps(candidates)})
        items = list(candidates.items())
        rng.shuffle(items)
        generated = {}
        intended = {}
        for index, (key, response) in enumerate(items):
            rating = int(key.removeprefix("rating_"))
            if not 1 <= rating <= 4:
                raise ValueError(f"Unexpected generated rating: {rating}")
            generated[f"ID_{index}"] = response
            intended[f"ID_{index}"] = rating
        schema = {
            key: {"rating": "<rating>", "reasoning": "<thinking>...</thinking>"}
            for key in generated
        }
        prompt = format_prompt(
            evaluation_template,
            **values,
            generated_responses=str(generated),
            output_format=str(schema),
        )
        evaluated = generate_json(
            [{"role": "user", "content": prompt}], evaluation_temperature, max_tokens
        )
        if set(evaluated) != set(generated):
            raise ValueError("Evaluator did not score every generated response")
        feedback = {}
        for key, intended_rating in intended.items():
            rating = int(evaluated[key]["rating"])
            reasoning = evaluated[key]["reasoning"]
            item = {
                "response": generated[key],
                "gen_rating": intended_rating,
                "eval_rating": rating,
                "eval_reasoning": reasoning,
            }
            if accepted_rating(intended_rating, rating, threshold):
                accepted[str(intended_rating)] = item
            else:
                feedback[f"rating_{intended_rating}"] = {
                    "response": generated[key],
                    "intended_rating": intended_rating,
                    "eval_rating": rating,
                    "eval_reasoning": reasoning,
                }
        history.append({"iteration": iteration, "feedback": feedback})
        if not feedback:
            break
        pending = set(feedback)
        prompt = format_prompt(
            feedback_template,
            **values,
            feedback_data=str(feedback),
            output_format=str({key: "[revised_response]" for key in feedback}),
        )
        messages.append({"role": "user", "content": prompt})
    return {"example": example, "accepted": accepted, "history": history}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap rating-controlled responses from descriptions"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--generation-temperature", type=float, default=0.6)
    parser.add_argument("--evaluation-temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.threshold < 0 or args.max_iterations < 0 or args.max_tokens <= 0:
        parser.error("Threshold and iterations must be nonnegative; max tokens must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    templates = [
        Path("prompts", name).read_text()
        for name in ["gen_prompt_desc.txt", "eval_prompt_desc.txt", "feedback_gen_prompt_desc.txt"]
    ]
    examples = load_records(args.dataset, args.split)
    if args.limit:
        examples = examples[: args.limit]
    if args.dry_run:
        if not examples:
            raise ValueError("No seed examples")
        row = examples[0]
        print(
            format_prompt(
                templates[0],
                instruction=row["instruction"],
                gold_standard_response=row["response"],
                video_description=row["video_description"],
            )
        )
        return
    if args.output.exists():
        parser.error("Choose a new output file to avoid regenerating completed data")
    output = {}
    for example in examples:
        key = str(example["id"])
        if key in output:
            raise ValueError(f"Duplicate seed ID: {key}")
        output[key] = bootstrap_example(
            example,
            random.Random(f"{args.seed}:{key}"),
            args.threshold,
            args.max_iterations,
            *templates,
            args.max_tokens,
            args.generation_temperature,
            args.evaluation_temperature,
        )
        write_json(args.output, output)


if __name__ == "__main__":
    main()
