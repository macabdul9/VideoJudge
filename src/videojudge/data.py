from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def load_records(source: str, split: str = "train", revision: str | None = None) -> list[dict]:
    path = Path(source)
    if path.is_file():
        text = path.read_text(encoding="utf-8")
        data = (
            [json.loads(line) for line in text.splitlines() if line.strip()]
            if path.suffix == ".jsonl"
            else json.loads(text)
        )
        return list(data.values()) if isinstance(data, dict) else data
    from datasets import load_dataset

    return list(load_dataset(source, split=split, revision=revision))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def format_prompt(template: str, **values: str) -> str:
    # Output examples contain literal braces as well as input placeholders.
    template = template.replace("{{", "{").replace("}}", "}")
    return re.sub(
        r"\{([A-Za-z_][A-Za-z0-9_]*)\}", lambda match: str(values.get(match[1], match[0])), template
    )
