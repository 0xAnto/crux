import json

import pytest

from crux.data.proofwriter import split_theory, to_records, to_test_records
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


def test_to_test_records_prefixes_the_ids_and_records_the_subset(tmp_path):
    (tmp_path / "meta-test.jsonl").write_text(json.dumps({**ROW, "maxD": 4}) + "\n")
    recs = to_test_records(tmp_path, "birds_electricity")
    assert [r["id"] for r in recs][0] == "pw-birds_electricity-T-OWA-D5-1-Q1"
    assert recs[0]["meta"]["subset"] == "birds_electricity" and recs[0]["meta"]["dataset_depth"] == 4
    assert to_test_records(tmp_path, "depth5")[0]["meta"]["dataset_depth"] == 5
    for r in recs:
        validate(r)


# A ParaRules row: the paraphrased `theory` has ". " where `sentences` has ";", and two spaces after a period.
PARA = {
    "id": "Nat-1", "maxD": 3,
    "theory": "Bob is nice and kind.  Bob seems big. Kind people who look big are red.",
    "sentences": {"sent1": "Bob is nice and kind; Bob seems big.", "sent2": "Kind people who look big are red."},
    "mappings": {"triple1": "sent1", "triple2": "sent1", "rule1": "sent2"},
    "triples": {"triple1": {"text": "Bob is nice."}, "triple2": {"text": "Bob is kind."}},
    "rules": {"rule1": {"text": "If someone is kind and big then they are red."}},
    "questions": {"Q1": {"question": "Bob is red.", "answer": "Unknown", "QDep": 2, "strategy": "proof"}},
}


def test_pararules_uses_the_paraphrased_sentences(tmp_path):
    assert split_theory(PARA) == (["Bob is nice and kind; Bob seems big."], ["Kind people who look big are red."])
    (tmp_path / "meta-test.jsonl").write_text(json.dumps(PARA) + "\n")
    (rec,) = to_test_records(tmp_path, "pararules")
    assert rec["state"] == "Bob is nice and kind; Bob seems big."
    assert rec["instruction"] == "Kind people who look big are red. Question: Bob is red."


@pytest.mark.parametrize("change", [
    {"theory": "Bob is nice. Kind people who look big are red."},  # a changed sentence
    {"theory": "Kind people who look big are red. Bob is nice and kind. Bob seems big."},  # reordered
    {"theory": PARA["theory"] + " Bob seems big."},  # a doubled sentence
    {"theory": "Bob is nice and kind.  Bob seems big."},  # a missing sentence
    {"mappings": {**PARA["mappings"], "rule1": "sent1"}},  # one sentence is a fact and a rule
    {"rules": {}},  # a sentence is neither
])
def test_pararules_rejects_sentences_that_differ_from_the_theory(change):
    with pytest.raises(ValueError):
        split_theory({**PARA, **change})
