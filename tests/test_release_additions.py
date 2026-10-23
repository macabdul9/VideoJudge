import argparse
import json
import random
import sys
from pathlib import Path

import pytest
from datasets import Dataset

from videojudge import seed
from videojudge.experiments import sweep_commands
from videojudge.prepare import training_record
from videojudge.prepare_benchmarks import add_descriptions, llava_records, meta_splits, vcg_records
from videojudge.run_benchmarks import benchmark_commands

ROOT = Path(__file__).resolve().parents[1]


def test_feedback_training_preserves_pair_orientation_and_reasoning(tmp_path):
    (tmp_path / "clip.mp4").touch()
    example = {
        "video_file": "clip.mp4",
        "instruction": "Compare",
        "response_a": "Worse",
        "response_b": "Better",
        "answer": "B",
        "preference_reasoning": "B describes the visible event.",
    }
    for random_seed in [1, 2, 42]:
        record = training_record(
            example,
            "pairwise",
            tmp_path,
            "A:{response_a} B:{response_b}",
            random.Random(random_seed),
            pairwise_reasoning_column="preference_reasoning",
        )
        assert record["conversations"][0]["value"] == "<video>\nA:Worse B:Better"
        assert record["conversations"][1]["value"] == (
            "<thinking>B describes the visible event.</thinking>\n<answer>B</answer>"
        )
    example["preference_reasoning"] = "<thinking>Already tagged</thinking>"
    result = training_record(
        example,
        "pairwise",
        tmp_path,
        "{instruction}",
        random.Random(42),
        pairwise_reasoning_column="preference_reasoning",
    )
    assert result["conversations"][1]["value"].count("<thinking>") == 1
    example["answer"] = "C"
    with pytest.raises(ValueError, match="A or B"):
        training_record(
            example,
            "pairwise",
            tmp_path,
            "{instruction}",
            random.Random(42),
            pairwise_reasoning_column="preference_reasoning",
        )


def test_seed_sampling_has_two_merge_shuffles_and_a_sampling_shuffle(tmp_path, monkeypatch):
    rows = []
    for index in range(12):
        (tmp_path / f"v{index}.mp4").touch()
        rows.append({"video_id": f"v{index}", "q": f"Instruction {index}", "a": f"Answer {index}"})
    (tmp_path / "vcg.json").write_text(json.dumps(rows))
    for name in ["plus.json", "chat.json"]:
        (tmp_path / name).write_text("[]")
    descriptions = {f"v{i}.mp4": {"description": f"Scene {i}"} for i in range(12)}
    (tmp_path / "descriptions.json").write_text(json.dumps(descriptions))
    output = tmp_path / "seed.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "seed",
            "--video-instruct",
            str(tmp_path / "vcg.json"),
            "--vcg-plus",
            str(tmp_path / "plus.json"),
            "--videochat2",
            str(tmp_path / "chat.json"),
            "--descriptions",
            str(tmp_path / "descriptions.json"),
            "--video-dir",
            str(tmp_path),
            "--num-samples",
            "5",
            "--output",
            str(output),
        ],
    )
    seed.main()
    indices = Dataset.from_dict({"index": list(range(12))})
    twice = indices.shuffle(seed=42).shuffle(seed=42)
    expected = twice.shuffle(seed=42).select(range(5))["index"]
    actual = [int(row["id"].rsplit("__", 1)[1]) for row in json.loads(output.read_text())]
    assert actual == expected
    assert actual != twice.select(range(5))["index"]
    assert all(isinstance(row["video_description"], str) for row in json.loads(output.read_text()))


def test_vcg_and_llava_seed_construction_matches_source_rules(tmp_path):
    (tmp_path / "clip.mp4").touch()
    rows = [
        {"video_name": "missing", "question_1": "Question", "answer": "Answer"},
        {"video_name": "clip", "question_1": "None", "question_2": "Fallback", "answer": "Answer"},
    ]
    records = vcg_records(rows, "Consistency", tmp_path)
    assert records[0]["id"] == "lmms-lab_VideoChatGPT_clip_Consistency__1"
    assert records[0]["instruction"] == "Fallback"
    llava = llava_records(
        [
            {"id": "missing", "conversations": []},
            {
                "id": "clip",
                "conversations": [
                    {"from": "human", "value": "<video><image> Task"},
                    {"from": "gpt", "value": "Answer"},
                ],
            },
        ],
        tmp_path,
    )
    assert llava[0]["id"] == "lmms-lab_LLaVA-Video-178K_clip__0"
    assert llava[0]["instruction"] == "Task"
    described = add_descriptions(llava, {"clip.mp4": {"description": "A scene", "model_name": "M"}})
    assert described[0]["video_description_model"] == "M"


def test_meta_splits_drop_partial_groups_and_match_shuffle_sequence():
    complete = [
        {"video_id": "one", "gen_rating": str(rating), "response": str(rating)}
        for rating in range(1, 6)
    ]
    partial = [
        {"video_id": "two", "gen_rating": str(rating), "response": str(rating)}
        for rating in range(1, 5)
    ]
    duplicate = {**complete[0], "response": "Duplicate must not replace first"}
    seeds = [{"id": "one", "instruction": "Task", "video_id": "one.mp4"}]
    splits = meta_splits(
        complete + partial + [duplicate], seeds, {"one.mp4": "Description"}, mini_size=3
    )
    assert len(splits["test"]) == 5
    assert len(splits["testmini"]) == 3
    one_shuffle = Dataset.from_list(complete).shuffle(seed=42)
    expected_test = one_shuffle.shuffle(seed=42)
    expected_mini = expected_test.select(range(3)).shuffle(seed=42)
    assert [row["response"] for row in splits["test"]] == expected_test["response"]
    assert [row["response"] for row in splits["testmini"]] == expected_mini["response"]
    assert len({row["unique_id"] for row in splits["test"]}) == 5


def sweep_args(kind, tmp_path):
    return argparse.Namespace(
        kind=kind,
        values=None,
        output_dir=tmp_path,
        model="checkpoint",
        data_config=Path("data/train/datasets.json"),
        data_config_pattern="data/{value}/datasets.json",
        dataset="examples.json",
        video_dir=Path("videos"),
        split="test",
        prompt=Path("prompt.txt"),
        fps=1,
        max_frames=180,
        temperature=0.5,
        backend="hf",
        tensor_parallel_size=1,
    )


def test_sweeps_separate_greedy_and_sampling_and_preserve_training_profile(tmp_path, monkeypatch):
    for name in ["NPROC_PER_NODE", "MICRO_BATCH_SIZE", "GRADIENT_ACCUMULATION_STEPS"]:
        monkeypatch.delenv(name, raising=False)
    grids = json.loads((ROOT / "configs/ablations.json").read_text())
    temperature = sweep_commands(sweep_args("temperature", tmp_path), grids)
    assert len(temperature) == 11
    assert "--do-sample" not in temperature[0][0]
    assert "--do-sample" in temperature[1][0]
    frames = sweep_commands(sweep_args("training-frames", tmp_path), grids)
    command, environment = frames[0]
    assert command[command.index("--video_max_frames") + 1] == "30"
    assert command[command.index("--num_train_epochs") + 1] == "3"
    assert environment["NPROC_PER_NODE"] == "2"
    assert environment["MICRO_BATCH_SIZE"] == "8"
    assert environment["GRADIENT_ACCUMULATION_STEPS"] == "1"


def test_benchmark_suite_forwards_columns_and_keeps_separate_outputs(tmp_path):
    configs = json.loads((ROOT / "configs/benchmarks.json").read_text())
    commands = benchmark_commands(
        "checkpoint",
        {"vatex": configs["vatex"], "longvideo": configs["longvideo"]},
        tmp_path,
        "hf",
        None,
    )
    assert commands[0][commands[0].index("--rating-column") + 1] == "score"
    assert commands[1][commands[1].index("--group-column") + 1] == "video_id,instruction"
    assert (
        commands[0][commands[0].index("--output-dir") + 1]
        != (commands[1][commands[1].index("--output-dir") + 1])
    )


def test_video_validation_rejects_empty_and_decodes_valid_clip(tmp_path):
    import av
    import numpy as np

    from videojudge.validate_videos import check_video

    empty = tmp_path / "empty.mp4"
    empty.touch()
    assert check_video(str(empty))["error"] is not None
    valid = tmp_path / "valid.mp4"
    with av.open(str(valid), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=1)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        frame = av.VideoFrame.from_ndarray(np.zeros((16, 16, 3), dtype=np.uint8), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    assert check_video(str(valid))["error"] is None
