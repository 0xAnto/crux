import copy

import pytest

from crux.schema import read_jsonl, validate, write_jsonl

GOOD = {
    "binary": {"id": "b1", "source": "ruletaker", "task_type": "binary", "state": "Anne is red.",
               "instruction": "Red things are big. Question: Anne is big.", "options": ["True", "False"],
               "target": 0, "meta": {"depth": 1, "label": "true"}},
    "multi_choice": {"id": "m1", "source": "synthetic", "task_type": "multi_choice", "state": "Bob has 3 coins.",
                     "instruction": "Who can pay 2 coins?", "options": ["Bob", "Carl", "Dana"], "target": 0,
                     "meta": {"template_id": "t7"}},
    "scoring": {"id": "s1", "source": "synthetic", "task_type": "scoring", "state": "x is 5.",
                "instruction": "Rate each option.", "options": ["a", "b"], "target": [0.25, 1.0], "meta": {}},
}


@pytest.mark.parametrize("task", GOOD)
def test_valid_records_pass(task):
    validate(GOOD[task])


@pytest.mark.parametrize("task, change, message", [
    ("binary", {"state": "  "}, "'state' must be a non-empty string"),
    ("binary", {"instruction": None}, "'instruction' must be a non-empty string"),
    ("binary", {"source": "Rule Taker"}, "'source' must match"),
    ("binary", {"task_type": "regression"}, "'task_type' must be one of"),
    ("binary", {"target": 2}, "out of range"),
    ("binary", {"target": True}, "int option index"),
    ("binary", {"options": ["True", "False", "Unknown", "Maybe"]}, "a binary task has 2 options"),
    ("binary", {"target": 1}, "meta 'label' is 'true'"),
    ("binary", {"options": ["True", "False", "Maybe"]}, "must be 'Unknown'"),
    ("multi_choice", {"options": ["Bob", " bob", "Dana"]}, "duplicate options"),
    ("multi_choice", {"options": ["Bob"]}, "2 or more items"),
    ("scoring", {"target": [0.5]}, "one score per option"),
    ("scoring", {"target": [0.5, float("nan")]}, "finite number"),
    ("multi_choice", {"meta": {"template_id": 7}}, "meta 'template_id' must be str"),
])
def test_bad_records_are_rejected(task, change, message):
    rec = {**copy.deepcopy(GOOD[task]), **change}
    with pytest.raises(ValueError, match=message):
        validate(rec)


def test_missing_field_is_rejected():
    rec = {k: v for k, v in GOOD["binary"].items() if k != "meta"}
    with pytest.raises(ValueError, match=r"missing fields \['meta'\]"):
        validate(rec)


def test_jsonl_round_trip_and_duplicate_ids(tmp_path):
    path = write_jsonl(GOOD.values(), tmp_path / "x" / "train.jsonl")
    assert read_jsonl(path) == list(GOOD.values())
    with pytest.raises(ValueError, match="duplicate ids"):
        write_jsonl([GOOD["binary"], GOOD["binary"]], tmp_path / "dupe.jsonl")
