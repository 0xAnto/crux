import pytest

from crux.data.ruletaker import load_split, parse_row

ROW = {
    "id": "RelNeg-D2-1",
    "context": "The bald eagle is nice. If something is nice then it is kind.",
    "questions": [
        {"id": "RelNeg-D2-1-1", "text": "The bald eagle is kind.", "label": True},
        {"id": "RelNeg-D2-1-2", "text": "The bald eagle is not nice.", "label": False},
    ],
}


def test_parse_row_builds_pairs_and_maps_labels():
    ex = parse_row(ROW, ["false", "true"])
    assert [e["label"] for e in ex] == [1, 0]
    assert ex[0]["context"] == ROW["context"]
    assert ex[1]["question"] == "The bald eagle is not nice."


def test_three_labels_from_the_label_list():
    row = {"context": "c", "questions": [{"id": "q", "text": "t", "label": "Unknown"}]}
    assert parse_row(row, ["false", "true", "unknown"])[0]["label"] == 2


def test_test_split_is_refused(tmp_path):
    with pytest.raises(ValueError):
        load_split(tmp_path, "test", ["false", "true"])


def test_to_records_puts_facts_in_state_and_rules_in_instruction(tmp_path):
    import json

    from crux.data.ruletaker import to_records
    from crux.schema import validate

    # The context order is rule1, triple2, triple1 (sentenceScramble gives positions in triples + rules).
    row = {"id": "T1", "context": "If something is nice then it is kind. Bob is red. Anne is nice.",
           "meta": {"sentenceScramble": [3, 2, 1]},
           "questions": [{"id": "T1-1", "text": "Anne is kind.", "label": True,
                          "meta": {"QDep": 1, "QLen": 2, "strategy": "proof", "Qid": "Q1"}}]}
    meta = {"id": "T1", "triples": {"triple1": {"text": "Anne is nice."}, "triple2": {"text": "Bob is red."}},
            "rules": {"rule1": {"text": "If something is nice then it is kind."}}}
    (tmp_path / "dev.jsonl").write_text(json.dumps(row) + "\n")
    (tmp_path / "meta-dev.jsonl").write_text(json.dumps(meta) + "\n")
    (r,) = to_records(tmp_path, "dev", depth=3)
    assert r["state"] == "Bob is red. Anne is nice."
    assert r["instruction"] == "If something is nice then it is kind. Question: Anne is kind."
    assert (r["options"], r["target"], r["meta"]["depth"], r["meta"]["dataset_depth"]) == (["True", "False"], 0, 1, 3)
    validate(r)
