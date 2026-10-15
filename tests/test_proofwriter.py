import json

import pytest

from crux.data.proofwriter import to_records
from crux.schema import validate

ROW = {
    "id": "T-OWA-D5-1",
    "theory": "Bob is red. Anne is nice. If someone is red then they are kind.",
    "triples": {"triple1": {"text": "Bob is red."}, "triple2": {"text": "Anne is nice."}},
    "rules": {"rule1": {"text": "If someone is red then they are kind."}},
    "questions": {
        "Q1": {"question": "Bob is kind.", "answer": True, "QDep": 1, "strategy": "proof"},
        "Q2": {"question": "Bob is not red.", "answer": False, "QDep": 0, "strategy": "inv-proof"},
        "Q3": {"question": "Anne is red.", "answer": "Unknown", "QDep": 0, "strategy": "random"},
    },
}


def test_to_records_splits_facts_and_rules_and_maps_all_labels(tmp_path):
    (tmp_path / "meta-dev.jsonl").write_text(json.dumps(ROW) + "\n")
    recs = to_records(tmp_path, "dev")
    assert [r["id"] for r in recs] == ["T-OWA-D5-1-Q1", "T-OWA-D5-1-Q2", "T-OWA-D5-1-Q3"]
    assert recs[0]["state"] == "Bob is red. Anne is nice."
    assert recs[0]["instruction"] == "If someone is red then they are kind. Question: Bob is kind."
    assert recs[0]["options"] == ["True", "False", "Unknown"]
    assert [r["target"] for r in recs] == [0, 1, 2]
    assert [r["meta"]["label"] for r in recs] == ["true", "false", "unknown"]
    assert recs[0]["meta"]["depth"] == 1 and recs[0]["meta"]["dataset_depth"] == 3
    for r in recs:
        validate(r)


def test_test_split_is_refused(tmp_path):
    with pytest.raises(ValueError):
        to_records(tmp_path, "test")
