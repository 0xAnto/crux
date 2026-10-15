import json

import pandas as pd
import pytest

from crux.data.dynabench import FILE, OPTIONS, to_records
from crux.schema import validate


def _write(tmp_path):
    meta = json.dumps({"num_rules": 2, "num_turns": 2, "domain": "retail"})
    rows = [
        {"policy": "1. Never use emojis.\n2. Be polite.", "transcript": "User: Hi\nAgent: Hello",
         "label": "PASS", "metadata": meta, "base_id": "a"},
        {"policy": "1. Never use emojis.\n2. Be polite.", "transcript": "User: Hi\nAgent: Go away",
         "label": "FAIL", "metadata": meta, "base_id": "b"},
    ]
    (tmp_path / FILE).parent.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(tmp_path / FILE)


def test_records_split_state_and_policy_and_map_labels(tmp_path):
    _write(tmp_path)
    recs = [validate(r) for r in to_records(tmp_path, "test")]
    assert [r["target"] for r in recs] == [0, 1]
    assert recs[0]["options"] == OPTIONS
    assert recs[0]["state"] == "User: Hi\nAgent: Hello"
    assert recs[0]["instruction"].startswith("1. Never use emojis.")
    assert recs[0]["instruction"].endswith("Question: Does the dialogue comply with the policy?")
    assert recs[0]["meta"]["eval_only"] and recs[1]["meta"]["label"] == "not compliant"
    assert recs[1]["meta"]["source_label"] == "FAIL"
    assert recs[0]["meta"]["policy_id"] == recs[1]["meta"]["policy_id"]


def test_only_the_test_split_is_allowed(tmp_path):
    with pytest.raises(ValueError):
        to_records(tmp_path, "train")
