from __future__ import annotations

import re


def extract_score(response: str) -> float | None:
    matches = re.findall(r"<score>\s*([0-9]+(?:\.[0-9]+)?)\s*</score>", response)
    if len(matches) != 1:
        return None
    score = float(matches[0])
    return score if 1 <= score <= 5 else None


def extract_answer(response: str) -> str | None:
    matches = re.findall(r"<answer>\s*([AB])\s*</answer>", response, re.IGNORECASE)
    return matches[0].upper() if len(matches) == 1 else None
