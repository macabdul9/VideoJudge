import ast
import json
import random
import runpy
import sys
from pathlib import Path

import pytest

from videojudge import bootstrap, evaluate
from videojudge.data import format_prompt
from videojudge.finalize import finalize_examples, preference_pairs
from videojudge.metrics import longvideo_metrics, pairwise_metrics, pointwise_metrics
from videojudge.parsing import extract_answer, extract_score
from videojudge.prepare import training_record

ROOT = Path(__file__).resolve().parents[1]


def test_score_comes_from_score_tag_and_invalid_answers_do_not_leak():
    assert extract_score("Reason 2026: compare scores 1 and 5. <score>4</score>") == 4
    assert extract_score("Rating is 4") is None
    assert extract_score("<score>99</score>") is None
    assert extract_score("<score>1</score><score>5</score>") is None
    assert extract_answer("<answer> b </answer>") == "B"
    assert extract_answer("No decision") is None


def test_prompt_literal_braces_and_input_braces_are_preserved():
    template = 'Instruction: {instruction}; Response: {response}; {{"rating_1": "example"}}'
    result = format_prompt(template, instruction="Keep {response} literal", response="{{code}}")
    assert (
        result
        == 'Instruction: Keep {response} literal; Response: {{code}}; {"rating_1": "example"}'
    )
    for path in (ROOT / "prompts").glob("*.txt"):
        format_prompt(path.read_text(), instruction="task", response="answer")


def test_training_records_keep_one_video_marker_and_randomize_pair_order(tmp_path):
    (tmp_path / "clip.mp4").touch()
    row = {
        "video_file": "clip.mp4",
        "instruction": "Describe",
        "response": "A dog",
        "gen_rating": 4,
        "eval_reasoning": "<thinking>Mostly right</thinking>",
    }
    result = training_record(
        row, "pointwise", tmp_path, "<video>\n{instruction} {response}", random.Random(42)
    )
    assert result["conversations"][0]["value"].count("<video>") == 1
    assert result["conversations"][1]["value"].endswith("<score>4</score>")
    row.update(selected_response="GOOD", rejected_response="BAD")
    orders = set()
    for seed in range(20):
        record = training_record(
            row, "pairwise", tmp_path, "A:{response_a} B:{response_b}", random.Random(seed)
        )
        prompt = record["conversations"][0]["value"]
        answer = extract_answer(record["conversations"][1]["value"])
        assert f"{answer}:GOOD" in prompt
        orders.add(answer)
    assert orders == {"A", "B"}


def test_bootstrap_refines_only_disagreements(monkeypatch):
    generation_count = 0
    evaluation_count = 0

    def fake_generate(messages, temperature, max_tokens):
        nonlocal generation_count, evaluation_count
        prompt = messages[-1]["content"]
        if prompt.startswith("EVAL "):
            evaluation_count += 1
            responses = ast.literal_eval(prompt.removeprefix("EVAL "))
            return {
                key: {
                    "rating": 3
                    if response == "response2" and evaluation_count == 1
                    else int(response[-1]),
                    "reasoning": "<thinking>Check</thinking>",
                }
                for key, response in responses.items()
            }
        generation_count += 1
        if generation_count == 1:
            return {f"rating_{i}": f"response{i}" for i in range(1, 5)}
        return {"rating_2": "response2"}

    monkeypatch.setattr(bootstrap, "generate_json", fake_generate)
    result = bootstrap.bootstrap_example(
        {"instruction": "Task", "response": "Gold", "video_description": "A video"},
        random.Random(42),
        0,
        5,
        "GENERATE",
        "EVAL {generated_responses}",
        "REFINE {feedback_data}",
        1000,
    )
    assert set(result["accepted"]) == {"1", "2", "3", "4"}
    assert len(result["history"]) == 2
    assert generation_count == evaluation_count == 2
    assert not bootstrap.accepted_rating(2, 3, 0)
    assert bootstrap.accepted_rating(2, 3, 1)


def test_complete_rating_groups_and_preference_sampling():
    example = {"video_file": "clip.mp4", "instruction": "Task", "response": "Gold"}
    accepted = {
        str(i): {"gen_rating": i, "response": str(i), "eval_reasoning": "Reason"}
        for i in range(1, 5)
    }
    output = {
        "one": {"example": example, "accepted": accepted},
        "incomplete": {"example": example, "accepted": {"1": accepted["1"]}},
    }
    rows = finalize_examples(output, {"one": {"score": 5, "reasoning": "Gold reason"}})
    assert len(rows) == 5
    pairs = preference_pairs(rows, 42)
    assert len(pairs) == 5
    assert pairs == preference_pairs(rows, 42)
    for pair in pairs:
        assert pair["selected_response"] == "Gold" or (
            int(pair["selected_response"]) > int(pair["rejected_response"])
        )


def test_metrics_keep_invalid_pairwise_outputs_in_accuracy_denominator():
    rows = [
        {"true_answer": "A", "eval_output": "<answer>A</answer>"},
        {"true_answer": "B", "eval_output": "invalid"},
    ]
    result = pairwise_metrics(rows)
    assert result["accuracy"] == 0.5
    assert result["accuracy_valid"] == 1
    rows = [
        {"true_rating": 1, "eval_output": "<score>2</score>"},
        {"true_rating": 5, "eval_output": "<score>4</score>"},
    ]
    result = pointwise_metrics(rows)
    assert result["rmse"] == result["mae"] == 1
    assert result["valid_outputs"] == 2
    assert pointwise_metrics([{"true_rating": 3, "eval_output": "invalid"}])["valid_outputs"] == 0


def test_longvideo_grouped_comparison_counts_ties_as_half():
    rows = [
        {"group_id": "q", "true_rating": 1, "eval_output": "<score>4</score>"},
        {"group_id": "q", "true_rating": 0, "eval_output": "<score>4</score>"},
        {"group_id": "q", "true_rating": 0, "eval_output": "<score>2</score>"},
    ]
    result = longvideo_metrics(rows)
    assert result["psup"] == 0.75
    assert result["delta_correct_distractor"] == 1


def test_qwen_dataset_registry_resolves_relative_annotations(tmp_path, monkeypatch):
    (tmp_path / "annotations.json").write_text("[]")
    registry = tmp_path / "datasets.json"
    registry.write_text(
        json.dumps({"pointwise": {"annotation_path": "annotations.json", "data_path": ""}})
    )
    monkeypatch.setenv("VIDEOJUDGE_DATA_CONFIG", str(registry))
    module = runpy.run_path(str(ROOT / "training/qwenvl/data/__init__.py"))
    config = module["data_list"](["pointwise%50"])[0]
    assert config["sampling_rate"] == 0.5
    assert config["annotation_path"] == str(tmp_path / "annotations.json")
    with pytest.raises(ValueError):
        module["data_list"](["absent"])


def test_evaluation_cpu_fixture_and_resume_config_guard(tmp_path, monkeypatch):
    (tmp_path / "clip.mp4").touch()
    dataset = tmp_path / "examples.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "unique_id": 7,
                    "instruction": "Task",
                    "response": "Answer",
                    "video_file": "clip.mp4",
                    "gen_rating": 4,
                }
            ]
        )
    )
    output = tmp_path / "output"
    command = [
        "evaluate",
        "--model",
        "fixture",
        "--dataset",
        str(dataset),
        "--video-dir",
        str(tmp_path),
        "--output-dir",
        str(output),
        "--prompt",
        str(ROOT / "prompts/evaluation_model.txt"),
    ]
    monkeypatch.setattr(sys, "argv", command)
    from videojudge import inference

    class FakeJudge:
        def __init__(self, *args):
            pass

        def generate(self, *args):
            return "Reason 2026 <score>4</score>"

    monkeypatch.setattr(inference, "QwenVideoJudge", FakeJudge)
    evaluate.main()
    predictions = json.loads((output / "predictions.json").read_text())
    assert predictions["7"]["predicted_rating"] == 4
    evaluate.main()
    monkeypatch.setattr(sys, "argv", [*command, "--max-frames", "60"])
    with pytest.raises(ValueError, match="another evaluation"):
        evaluate.main()


def test_seed_deduplication_is_scoped_to_each_video():
    from videojudge.seed import deduplicate

    rows = [
        {"video_file": "one.mp4", "instruction": "Describe the running dog"},
        {"video_file": "one.mp4", "instruction": "Describe the running dog"},
        {"video_file": "two.mp4", "instruction": "Describe the running dog"},
    ]
    assert deduplicate(rows) == [rows[0], rows[2]]
