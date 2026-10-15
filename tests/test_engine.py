import copy
import json
from pathlib import Path

import pytest

from crux.gen import engine

CASES = json.loads((Path(__file__).parent / "data" / "engine_cases.json").read_text())["cases"]
LIST_KEYS = ("facts", "rules", "exceptions", "priorities")


def reversed_lists(theory):
    return {k: v[::-1] if k in LIST_KEYS else v for k, v in theory.items()}


def check_case(case, theory):
    expect = case["expect"]
    if "error" in expect:
        errors = engine.validate(theory, case["query"])
        assert any(e.startswith(expect["error"] + ":") for e in errors), errors
        with pytest.raises(ValueError):
            engine.evaluate(theory, case["query"])
    else:
        assert engine.validate(theory, case["query"]) == []
        result = engine.evaluate(theory, case["query"])
        assert (result["label"], result["depth"]) == (expect["label"], expect["depth"])
        assert engine.features(theory) == expect["features"]


def test_case_file_has_all_cases():
    assert len(CASES) >= 51 and len({c["id"] for c in CASES}) == len(CASES)


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_shared_case(case):
    before = copy.deepcopy(case)
    check_case(case, case["theory"])
    assert case == before  # the engine does not change its arguments


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_shared_case_reversed_lists(case):
    check_case(case, reversed_lists(case["theory"]))


# Paths that the shared cases do not cover. The base theory is valid: a0 (int), a1, a2 -> a3.
BASE = {"entities": ["e0", "e1"], "attributes": {"a0": "int", "a1": "bool", "a2": "bool", "a3": "bool"},
        "facts": [{"entity": "e0", "attr": "a0", "value": 5}, {"entity": "e0", "attr": "a1", "value": True}],
        "rules": [{"id": "r0", "if": [{"attr": "a0", "op": ">=", "value": 5}], "then": {"attr": "a3", "neg": False}},
                  {"id": "r1", "if": [{"attr": "a1", "neg": False}], "then": {"attr": "a3", "neg": True}}],
        "exceptions": [], "priorities": [{"higher": "r1", "lower": "r0"}]}
QUERY = {"type": "binary", "entity": "e0", "attr": "a3", "neg": False}


def test_base_theory_is_valid():
    assert engine.validate(BASE) == []
    assert engine.evaluate(BASE, QUERY) == {"label": "False", "depth": 1}
    assert engine.composition_id(BASE) == "c20"  # thr (4) + pri (16)


@pytest.mark.parametrize("path, value, code", [
    (("extra",), 1, "schema"),
    (("entities",), [], "schema"),
    (("entities",), ["e0", "e0"], "schema"),
    (("entities",), ["E0"], "schema"),
    (("attributes", "a1"), "float", "schema"),
    (("facts", 0, "value"), 100, "schema"),
    (("facts", 1), {"entity": "e0", "attr": "a0", "value": 3}, "schema"),
    (("rules", 0, "if"), [], "schema"),
    (("rules", 0, "if", 0, "op"), "!=", "schema"),
    (("rules", 0, "if"), [{"attr": "a0", "op": ">", "value": 1}, {"attr": "a0", "op": "<", "value": 9}], "schema"),
    (("rules", 1, "id"), "r0", "schema"),
    (("facts", 0, "entity"), "e7", "ref"),
    (("rules", 1, "if", 0, "attr"), "a9", "ref"),
    (("priorities", 0, "lower"), "r9", "ref"),
    (("facts", 0, "value"), True, "type"),
    (("facts", 1, "value"), 1, "type"),
    (("rules", 0, "if", 0), {"attr": "a1", "op": ">", "value": 1}, "type"),
    (("rules", 0, "then", "attr"), "a0", "type"),
    (("priorities", 0, "lower"), "r1", "priority"),
    (("priorities",), [{"higher": "r1", "lower": "r0"}, {"higher": "r1", "lower": "r0"}], "priority"),
])
def test_static_error(path, value, code):
    theory = copy.deepcopy(BASE)
    node = theory
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    errors = engine.validate(theory, QUERY)
    assert any(e.startswith(code + ":") for e in errors), errors


@pytest.mark.parametrize("query", [
    {"type": "binary", "entity": "e9", "attr": "a3", "neg": False},
    {"type": "binary", "entity": "e0", "attr": "a9", "neg": False},
    {"type": "binary", "entity": "e0", "attr": "a3", "neg": 0},
    {"type": "binary", "entity": "e0", "attr": "a3"},
    {"type": "multi_choice", "entity": "e0", "candidates": ["a3"]},
    {"type": "multi_choice", "entity": "e0", "candidates": ["a1", "a1"]},
    {"type": "multi_choice", "entity": "e0", "candidates": ["a1", "a2", "a3", "a1", "a2", "a3", "a1", "a2", "a3"]},
])
def test_bad_query(query):
    errors = engine.validate(BASE, query)
    assert errors and errors[0].startswith("query:"), errors


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_check_cli(tmp_path, capsys):
    theory = BASE
    meta = {"depth": 1, "rule_types": ["thr", "pri"], "composition_id": "c20", "pair_role": "orig",
            "theory_hash": engine.theory_hash(theory), "query": QUERY}
    good = {"id": "g", "options": ["True", "False", "Unknown"], "target": 1, "meta": meta}
    wrong_label = {**good, "id": "l", "target": 0}
    wrong_depth = {**good, "id": "d", "meta": {**meta, "depth": 2}}
    mc_query = {"type": "multi_choice", "entity": "e1", "candidates": ["a1", "a3"]}  # e1 has no true candidate
    invalid = {**good, "id": "m", "target": 0, "meta": {**meta, "query": mc_query}}
    missing = {**good, "id": "x", "meta": {**meta, "theory_hash": "0" * 16}}
    write_jsonl(tmp_path / "all.jsonl", [good, wrong_label, wrong_depth, invalid, missing])
    write_jsonl(tmp_path / "theories.jsonl", [{"theory": theory, "theory_hash": meta["theory_hash"]}])

    assert engine.main(["check", str(tmp_path / "all.jsonl")]) == 1
    out = capsys.readouterr().out
    for line in ("agree\t1", "disagree_label\t1\tl", "disagree_depth\t1\td", "invalid:mc_answer\t1\tm",
                 "missing_theory\t1\tx"):
        assert line in out.splitlines(), out
