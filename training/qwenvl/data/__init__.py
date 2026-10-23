"""Dataset configuration for the Qwen fine-tuning data loader."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path


def data_list(dataset_names: list[str]) -> list[dict]:
    registry = Path(os.environ.get("VIDEOJUDGE_DATA_CONFIG", "data/train/datasets.json"))
    datasets = json.loads(registry.read_text())
    configs = []
    for name in dataset_names:
        match = re.search(r"%(\d+)$", name)
        sampling_rate = int(match[1]) / 100.0 if match else 1.0
        key = re.sub(r"%(\d+)$", "", name)
        if key not in datasets:
            raise ValueError(f"Dataset {key!r} is not registered in {registry}")
        config = dict(datasets[key])
        annotation = Path(config["annotation_path"])
        if not annotation.is_absolute():
            annotation = registry.resolve().parent / annotation
        if not annotation.is_file():
            raise FileNotFoundError(annotation)
        config["annotation_path"] = str(annotation)
        config["sampling_rate"] = sampling_rate
        configs.append(config)
    return configs
