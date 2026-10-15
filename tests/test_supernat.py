import pytest

from crux.data.supernat import select_task, to_records
from crux.schema import validate


def make_task(categories, license_, outputs):
    return {
        "Categories": categories, "Instance License": license_, "Definition": ["Say if the text is positive."],
        "Positive Examples": [{"input": "few-shot input", "output": "Yes"}],
        "Instances": [{"id": f"task9-{i}", "input": f"text {i}", "output": [o]} for i, o in enumerate(outputs)],
    }


YES_NO = ["Yes", "No"] * 30


def test_selection_accepts_classification_and_rejects_others():
    labels, _ = select_task(make_task(["Textual Entailment"], ["MIT"], YES_NO))
    assert labels == ["No", "Yes"]
    assert select_task(make_task(["Title Generation"], ["MIT"], YES_NO))[1] == "category"
    assert select_task(make_task(["Textual Entailment"], ["Unknown"], YES_NO))[1] == "license_unknown"
    assert select_task(make_task(["Textual Entailment"], ["MIT"], [f"t{i}" for i in range(60)]))[1] == "label_set"


def test_to_records_maps_output_to_option_index(monkeypatch):
    task = make_task(["Answerability Classification"], ["CC BY-NC 4.0"], YES_NO)
    recs = to_records(None, "test", loaded=[("task9_x", task)])
    assert len(recs) == 60
    r = recs[0]  # output "Yes" -> index 1 in the sorted options
    assert r["options"] == ["No", "Yes"] and r["target"] == 1 and r["task_type"] == "multi_choice"
    assert r["state"] == "text 0" and "few-shot" not in r["instruction"]
    assert r["instruction"].startswith("Say if the text is positive.") and "Question:" in r["instruction"]
    assert r["meta"]["task_id"] == "task9_x" and r["meta"]["eval_only"] and r["meta"]["label"] == "Yes"
    validate(r)


def test_other_splits_are_refused():
    with pytest.raises(ValueError):
        to_records(None, "train", loaded=[])
