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
