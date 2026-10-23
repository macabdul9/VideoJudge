from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from videojudge.data import write_json
from videojudge.parsing import extract_answer, extract_score


def pointwise_metrics(records: list[dict], bins: int = 10) -> dict:
    valid = [
        (float(row["true_rating"]), extract_score(row["eval_output"]))
        for row in records
        if row.get("true_rating") is not None
    ]
    valid = [(true, predicted) for true, predicted in valid if predicted is not None]
    result = {"total_outputs": len(records), "valid_outputs": len(valid)}
    if not valid:
        return result
    true, predicted = np.asarray(valid, dtype=float).T
    differences = predicted - true
    result.update(
        rmse=float(np.sqrt(np.mean(differences**2))), mae=float(np.mean(np.abs(differences)))
    )
    if len(valid) > 1 and np.ptp(true) > 0 and np.ptp(predicted) > 0:
        result.update(
            pearson=float(pearsonr(true, predicted).statistic),
            spearman=float(spearmanr(true, predicted).statistic),
        )
    else:
        result.update(pearson=None, spearman=None)
    # Preserve the experiment script's half-open calibration bins, including its
    # exclusion of predictions equal to the maximum bin edge.
    edges = np.linspace(predicted.min(), predicted.max(), bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (predicted >= lower) & (predicted < upper)
        if mask.any():
            ece += float(mask.mean() * abs(predicted[mask].mean() - true[mask].mean()))
    result["ece"] = ece
    return result


def pairwise_metrics(records: list[dict]) -> dict:
    labelled = [row for row in records if row.get("true_answer") is not None]
    predictions = [
        (str(row["true_answer"]).strip().upper(), extract_answer(row["eval_output"]))
        for row in labelled
    ]
    if any(label not in {"A", "B"} for label, _ in predictions):
        raise ValueError("Pairwise labels must be A or B")
    correct = sum(label == predicted for label, predicted in predictions)
    valid = sum(predicted is not None for _, predicted in predictions)
    return {
        "total_outputs": len(records),
        "labelled_outputs": len(labelled),
        "valid_outputs": valid,
        "correct": correct,
        "accuracy": correct / len(labelled) if labelled else None,
        "accuracy_valid": correct / valid if valid else None,
    }


def longvideo_metrics(records: list[dict]) -> dict:
    groups = {}
    valid = 0
    for row in records:
        if "group_id" not in row:
            raise ValueError(
                "LongVideoBench predictions need group_id; evaluate with --group-column"
            )
        score = extract_score(row["eval_output"])
        if score is None:
            continue
        label = int(row["true_rating"])
        if label not in {0, 1}:
            raise ValueError("LongVideoBench labels must be 0 or 1")
        groups.setdefault(str(row["group_id"]), {0: [], 1: []})[label].append(score)
        valid += 1
    gaps, superiorities = [], []
    for group in groups.values():
        if not group[0] or not group[1]:
            continue
        correct, distractors = np.asarray(group[1]), np.asarray(group[0])
        comparisons = correct[:, None] - distractors[None, :]
        gaps.append(float(correct.mean() - distractors.mean()))
        superiorities.append(float(np.mean((comparisons > 0) + 0.5 * (comparisons == 0))))
    return {
        "total_outputs": len(records),
        "valid_outputs": valid,
        "complete_groups": len(gaps),
        "delta_correct_distractor": float(np.mean(gaps)) if gaps else None,
        "psup": float(np.mean(superiorities)) if superiorities else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize VideoJudge predictions")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=["pointwise", "pairwise", "longvideo"], default="pointwise"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ece-bins", type=int, default=10)
    args = parser.parse_args()
    if args.ece_bins <= 0:
        parser.error("--ece-bins must be positive")
    data = json.loads(args.predictions.read_text())
    records = list(data.values()) if isinstance(data, dict) else data
    if args.mode == "pointwise":
        metrics = pointwise_metrics(records, args.ece_bins)
    elif args.mode == "pairwise":
        metrics = pairwise_metrics(records)
    else:
        metrics = longvideo_metrics(records)
    write_json(args.output, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
