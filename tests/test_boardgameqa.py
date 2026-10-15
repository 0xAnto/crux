import json

import pytest

from crux.data.boardgameqa import OPTIONS, QUESTION_MARKER, to_records
from crux.schema import validate

FACTS = "The cat has a ball."
RULES = "Rule1: If the cat has a ball, then the cat plays."
PREFS = "Rule1 is preferred over Rule2."


def _row(label, prefs=PREFS):
    parts = ["A few players are playing a boardgame.", FACTS, RULES, prefs]
    return {"facts": FACTS, "rules": RULES, "preferences": prefs, "label": label,
            "example": " ".join(p for p in parts if p) + " " + QUESTION_MARKER + "does the cat play?"}


def test_to_records_splits_fields_and_maps_labels(tmp_path):
    (tmp_path / "BoardgameQA-Main-depth2").mkdir()
    rows = [_row("proved"), _row("disproved", ""), _row("unknown")]
    (tmp_path / "BoardgameQA-Main-depth2" / "valid.json").write_text(json.dumps(rows))
    recs = to_records(tmp_path, "valid", "BoardgameQA-Main-depth2")
    assert [r["target"] for r in recs] == [OPTIONS.index(x) for x in ("proved", "disproved", "unknown")]
    assert recs[0]["state"] == FACTS
    assert recs[0]["instruction"] == f"{RULES} {PREFS} Question: does the cat play?"
    assert recs[1]["instruction"] == f"{RULES} Question: does the cat play?"
    assert recs[0]["meta"]["depth"] == 2 and recs[0]["meta"]["evaluation_only"]
    for r in recs:
        validate(r)


def test_binary_subset_has_two_options(tmp_path):
    (tmp_path / "BoardgameQA-Binary-depth1").mkdir()
    rows = [_row("proved"), _row("disproved")]
    (tmp_path / "BoardgameQA-Binary-depth1" / "test.json").write_text(json.dumps(rows))
    recs = to_records(tmp_path, "test", "BoardgameQA-Binary-depth1")
    assert all(r["options"] == ["proved", "disproved"] for r in recs)
    assert [r["target"] for r in recs] == [0, 1]
    for r in recs:
        validate(r)


def test_bad_question_is_rejected(tmp_path):
    (tmp_path / "BoardgameQA-Main-depth1").mkdir()
    bad = _row("proved")
    bad["example"] = "no marker here"
    (tmp_path / "BoardgameQA-Main-depth1" / "test.json").write_text(json.dumps([bad]))
    with pytest.raises(ValueError):
        to_records(tmp_path, "test", "BoardgameQA-Main-depth1")
