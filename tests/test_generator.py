"""Tests of the Step 3 generator (docs/generator_spec.md, sections 4 to 11). No network, no engine."""
import copy
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import pytest
import yaml

from crux.gen import generator as G
from crux.gen.templates import FAMILIES, OPS, group
from crux.schema import validate

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / "tests" / "data" / "engine_cases.json").read_text())["cases"]
SEED = 20261015
META_KEYS = ["template_id", "template_group", "composition_id", "vocabulary_id", "block", "depth", "unit_depth", "label",
             "unit_id", "pair_id", "pair_role", "edit_type", "edit_path", "label_flip", "edit_relevant", "paraphrase_group",
             "rule_types",
             "theory_hash", "query", "attempt"]


def small_config(total=600):
    cfg = yaml.safe_load(open(ROOT / "configs" / "step3_generator.yaml"))
    cfg.update(total_records=total, engine_check=False, tokenizer_check=False)
    return cfg


def outcome(theory, query):
    try:
        return G.solve(theory, query)
    except ValueError as e:
        return str(e).split(":")[0]


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_solve_passes_shared_case(case):
    expect = case["expect"]
    if "error" in expect:
        with pytest.raises(ValueError, match=f"^{expect['error']}: "):
            G.solve(case["theory"], case["query"])
    else:
        assert G.solve(case["theory"], case["query"]) == {"label": expect["label"], "depth": expect["depth"]}
        assert G.features(case["theory"]) == expect["features"]


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_solve_ignores_list_order(case):
    rev = {**case["theory"], **{k: case["theory"][k][::-1] for k in ("facts", "rules", "exceptions", "priorities")}}
    assert outcome(rev, case["query"]) == outcome(case["theory"], case["query"])


def test_solve_rejects_ids_that_are_not_strings():
    theory = {"entities": ["e0"], "attributes": {"a0": "bool", "a1": "bool", "a2": "bool"},
              "facts": [{"entity": "e0", "attr": "a0", "value": True}],
              "rules": [{"id": "r0", "if": [{"attr": "a0", "neg": False}], "then": {"attr": "a1", "neg": False}},
                        {"id": "r1", "if": [{"attr": "a0", "neg": False}], "then": {"attr": "a1", "neg": True}}],
              "exceptions": [{"id": "x0", "if": [{"attr": "a2", "neg": False}], "blocks": "r0"}],
              "priorities": [{"higher": "r0", "lower": "r1"}]}
    query = {"type": "binary", "entity": "e0", "attr": "a1", "neg": False}
    assert G.solve(theory, query)["label"] == "True"
    for path in (("exceptions", 0, "blocks"), ("facts", 0, "entity"), ("rules", 0, "then", "attr"),
                 ("priorities", 0, "higher"), ("rules", 0, "if", 0, "attr")):
        bad = copy.deepcopy(theory)
        doc = bad
        for key in path[:-1]:
            doc = doc[key]
        doc[path[-1]] = [doc[path[-1]]]
        with pytest.raises(ValueError):
            G.solve(bad, query)
    with pytest.raises(ValueError):
        G.solve(theory, {**query, "entity": ["e0"]})


@pytest.fixture(scope="module")
def run600(tmp_path_factory):
    out = tmp_path_factory.mktemp("gen600")
    report = G.run(small_config(), SEED, out)
    records = [json.loads(line) for line in open(out / "all.jsonl")]
    theories = {}
    for line in open(out / "theories.jsonl"):
        row = json.loads(line)
        theories[row["theory_hash"]] = row["theory"]
    return out, report, records, theories


def test_same_seed_gives_same_bytes_and_other_seed_differs(run600, tmp_path):
    out = run600[0]
    G.run(small_config(), SEED, tmp_path / "same")
    G.run(small_config(), SEED + 1, tmp_path / "other")
    for name in ("all.jsonl", "theories.jsonl", "report.json"):
        assert (out / name).read_bytes() == (tmp_path / "same" / name).read_bytes(), name
    assert (out / "all.jsonl").read_bytes() != (tmp_path / "other" / "all.jsonl").read_bytes()


def test_records_pass_the_schema_and_carry_the_meta_keys(run600):
    _, report, records, _ = run600
    assert len(records) == 600 == report["counts"]["records"]
    assert len({r["id"] for r in records}) == 600
    for r in records:
        validate(r)
        assert r["source"] == "synthetic"
        assert list(r["meta"]) == META_KEYS
        assert r["id"] == f"syn-{r['meta']['unit_id']}-{r['meta']['template_id']}-{r['meta']['pair_role']}"
        assert r["instruction"].count("Question:") == 1 and r["instruction"].endswith("?")


def test_labels_depths_and_features_come_from_the_theory(run600):
    _, _, records, theories = run600
    for r in records:
        m = r["meta"]
        theory = theories[m["theory_hash"]]
        assert G.theory_hash(theory) == m["theory_hash"]
        assert G.solve(theory, m["query"]) == {"label": m["query"]["candidates"][r["target"]]
                                               if r["task_type"] == "multi_choice" else m["label"], "depth": m["depth"]}
        assert m["rule_types"] == G.features(theory)
        if m["pair_role"] == "orig":
            assert m["depth"] == m["unit_depth"]
            assert G.comp_features(m["composition_id"]) == m["rule_types"]


def _get(doc, path):
    for part in path.strip("/").split("/"):
        doc = doc[int(part)] if isinstance(doc, list) else doc[part]
    return doc


def test_pair_members_differ_by_exactly_one_edit(run600):
    _, _, records, theories = run600
    pairs = defaultdict(dict)
    for r in records:
        pairs[r["meta"]["pair_id"]][r["meta"]["pair_role"]] = r
    assert pairs
    for pair in pairs.values():
        o, e = pair["orig"]["meta"], pair["edit"]["meta"]
        for key in ("template_id", "composition_id", "vocabulary_id", "block", "edit_type", "edit_path", "label_flip",
                    "edit_relevant", "unit_depth", "query"):
            assert o[key] == e[key]
        assert o["label_flip"] == (o["label"] != e["label"])
        assert o["edit_relevant"] or not o["label_flip"]
        assert (pair["orig"]["state"], pair["orig"]["instruction"]) != (pair["edit"]["state"], pair["edit"]["instruction"])
        t, t2, path, et = theories[o["theory_hash"]], theories[e["theory_hash"]], o["edit_path"], o["edit_type"]
        expect = copy.deepcopy(t)
        *head, last = path.strip("/").split("/")
        parent = _get(expect, "/".join(head))
        if et in ("fact_remove", "exception_remove", "rule_remove"):
            del parent[int(last)]
        elif et == "priority_swap":
            p = parent[int(last)]
            parent[int(last)] = {"higher": p["lower"], "lower": p["higher"]}
        elif et == "fact_flip" or (et == "condition_negate" and last == "neg"):
            parent[last] = not parent[last]
        elif et == "condition_negate":
            parent[last] = G.COMPLEMENT[parent[last]]
        else:  # fact_number, threshold_shift: a new integer 0 to 99
            new = _get(t2, path)
            assert isinstance(new, int) and 0 <= new <= 99 and new != parent[last]
            parent[last] = new
        assert expect == t2, path


def test_paraphrases_share_theory_and_label(run600):
    _, _, records, _ = run600
    groups = defaultdict(list)
    for r in records:
        groups[r["meta"]["paraphrase_group"]].append(r)
    assert any(len(g) > 1 for g in groups.values())
    for g in groups.values():
        for key in ("theory_hash", "query", "label", "template_group"):
            assert len({json.dumps(r["meta"][key]) for r in g}) == 1
        assert len({(r["state"], json.dumps(r["options"]), r["target"]) for r in g}) == 1
        assert len({r["instruction"] for r in g}) == len(g)
        assert len({r["meta"]["template_id"] for r in g}) == len(g)


def test_no_held_out_id_occurs_in_the_train_block(run600):
    _, _, records, _ = run600
    blocks = {r["meta"]["block"] for r in records}
    assert blocks == {"train", "val", "test"}
    for r in records:
        m = r["meta"]
        assert m["composition_id"] in G.COMPOSITIONS[m["block"]]
        assert m["vocabulary_id"] in G.VOCAB_IDS[m["block"]]
        assert m["template_group"] == group(m["template_id"]) in G.TEMPLATE_GROUPS[m["block"]]
        if m["block"] == "train":
            assert m["depth"] <= 3 and m["unit_depth"] <= 3
    held = set(G.HELD_OUT["val"] + G.HELD_OUT["test"])
    assert len(held) == 7 and not held & set(G.COMPOSITIONS["train"])


def test_multi_choice_has_exactly_one_true_option(run600):
    _, _, records, theories = run600
    mc = [r for r in records if r["task_type"] == "multi_choice"]
    assert mc
    for r in mc:
        theory, query = theories[r["meta"]["theory_hash"]], r["meta"]["query"]
        assert 2 <= len(r["options"]) <= 8 and len(r["options"]) == len(query["candidates"])
        val, _ = G._model(theory, G._check_theory(theory))
        true = [i for i, c in enumerate(query["candidates"]) if val.get((query["entity"], c)) is True]
        assert true == [r["target"]]


def test_both_polarities_occur_at_the_same_rate_for_each_label(run600):
    """Orig records only: a rule_remove edit can remove one polarity (the audit check polarity.edit covers the edit
    records at 100K)."""
    _, _, records, theories = run600
    both = defaultdict(list)
    for r in records:
        m = r["meta"]
        if r["task_type"] == "binary" and m["depth"] >= 1 and m["pair_role"] == "orig":
            t, a = theories[m["theory_hash"]], m["query"]["attr"]
            both[m["label"]].append({x["then"]["neg"] for x in t["rules"] if x["then"]["attr"] == a} == {False, True})
    shares = {k: sum(v) / len(v) for k, v in both.items()}
    assert len(shares) == 3 and max(shares.values()) - min(shares.values()) <= 0.05, shares


def _content_tokens(pool):  # spec: tokens of the lowercased string values (not the keys); placeholders use digits
    def strings(x):
        if isinstance(x, str):
            yield x
        elif isinstance(x, dict):
            for v in x.values():
                yield from strings(v)
        elif isinstance(x, list):
            for v in x:
                yield from strings(v)
    return {w for s in strings(pool) for w in re.findall(r"[a-z]+", s.lower())} - G.STOP


def test_vocabulary_pools_are_disjoint_and_avoid_the_denylist():
    deny = {line.strip() for line in open(G.DENYLIST) if line.strip() and not line.startswith("#")}
    assert sorted(G.VOCAB) == ["v%02d" % i for i in range(12)]
    tokens = {}
    for vid, pool in G.VOCAB.items():
        assert len(pool["names"]) >= 12 and len(pool["bool"]) >= 24 and len(pool["int"]) >= 4
        assert pool["kind"][0] not in "aeiou"  # the frames write "a {kind}"
        phrases = pool["names"] + [p for pair in pool["bool"] for p in pair] + [d[k] for d in pool["int"] for k in d]
        assert len(phrases) == len(set(phrases)), vid
        tokens[vid] = _content_tokens(pool)
        assert not tokens[vid] & deny, (vid, sorted(tokens[vid] & deny))
    ids = sorted(tokens)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            assert not tokens[a] & tokens[b], (a, b, sorted(tokens[a] & tokens[b]))


FORBIDDEN = ["rule", "rules", "preferred", "preference", "conclude", "conclusion", "regarding", "game", "player",
             "at least one", "be certain", "positive that", "in the case where", "whenever", "surely", "unquestionably",
             "for sure", "careful", "problematic", "belief", "something", "someone", "things", "people"]


def test_template_families_use_their_own_frames():
    assert list(FAMILIES) == ["t%02d" % n for n in range(15)]
    frames = defaultdict(set)
    for tid, fam in FAMILIES.items():
        for slot, frame in fam.items():
            for key, text in (frame.items() if isinstance(frame, dict) else [("", frame)]):
                assert text not in frames[(slot, key)], (tid, slot, key)
                frames[(slot, key)].add(text)
                low = text.lower()
                assert "question:" not in low and not re.search(r"rule\s*\d", low)
                for bad in FORBIDDEN:
                    assert not re.search(r"\b" + re.escape(bad) + r"\b", low), (tid, slot, bad)
        assert len({G.rule_label(tid, i) for i in range(20)}) == 20


def test_every_family_renders_every_form():
    theory = {"entities": ["e0", "e1"], "attributes": {"a0": "bool", "a1": "int", "a2": "bool", "a3": "bool", "a4": "bool"},
              "facts": [{"entity": "e0", "attr": "a0", "value": True}, {"entity": "e0", "attr": "a1", "value": 8},
                        {"entity": "e1", "attr": "a0", "value": False}],
              "rules": [{"id": "r%d" % i, "if": [{"attr": "a0", "neg": i == 1}, {"attr": "a1", "op": op, "value": 5}],
                         "then": {"attr": "a2", "neg": i % 2 == 1}} for i, op in enumerate(OPS)]
              + [{"id": "r5", "if": [{"attr": "a0", "neg": False}, {"attr": "a2", "neg": True}, {"attr": "a1", "op": ">", "value": 1}],
                  "then": {"attr": "a3", "neg": False}}],
              "exceptions": [{"id": "x0", "if": [{"attr": "a4", "neg": False}], "blocks": "r0"}],
              "priorities": [{"higher": "r0", "lower": "r1"}]}
    queries = [{"type": "binary", "entity": "e0", "attr": "a3", "neg": False},
               {"type": "binary", "entity": "e0", "attr": "a3", "neg": True},
               {"type": "multi_choice", "entity": "e1", "candidates": ["a2", "a3", "a4"]}]
    lex = G.make_lexicon(theory, "v00", random.Random(0))
    for tid in FAMILIES:
        texts = set()
        for q in queries:
            state, instruction, options = G.render(theory, q, lex, "v00", tid)
            assert instruction.count("Question: ") == 1 and instruction.endswith("?")
            assert all(s.endswith(".") for s in re.split(r"(?<=\.) (?=[A-Z])", state))
            assert "{" not in instruction and "}" not in instruction
            texts.add(instruction)
        assert len(texts) == 3


def test_tokenizer_does_not_truncate_a_sample_of_2000_records():
    records = [r for _, _, r in G.generate(small_config(2000), SEED)]
    counts = G.tokenizer_check(records, SEED)
    if counts is None:
        pytest.skip("the tokenizer of configs/step2_data.yaml is not in the local cache")
    assert counts["sample"] == 2000 and counts["joint_options"] == 0 and counts["separate"] == 0


def test_quota_prefers_the_edit_key_with_room(run600):
    # A flip key with many flips and no stays has no room, so choose_edit takes a flip with another key if it can.
    _, _, records, theories = run600
    moved = 0
    for r in records:
        m = r["meta"]
        if m["pair_role"] != "orig" or m["depth"] == 0:
            continue
        t, q = theories[m["theory_hash"]], m["query"]
        mine, feats = G.solve(t, q), G.features(t)
        free = G.choose_edit(t, q, mine, m["edit_type"], True, feats, random.Random(0))
        if free is None or free[3]["label"] == mine["label"]:
            continue
        full = {free[5]: [0, 99], G._coarse(free[5]): [0, 99]}
        got = G.choose_edit(t, q, mine, m["edit_type"], True, feats, random.Random(0), quota=full)
        moved += got[5] != free[5]
    assert moved > 0



def test_walk_keeps_the_answer_and_draws_free_facts_with_the_prior():
    """a0 -> a1 (query a1, True at depth 1) and a2 >= 50 -> a3. The walk must keep a0 true. a2 is free, so after the
    walk it is missing in about 1/4 of the runs (FACT_SHARES), although every run starts with a2 = 7."""
    import random
    query = {"type": "binary", "entity": "e0", "attr": "a1", "neg": False}
    missing, n = 0, 400
    for i in range(n):
        b = object.__new__(G._Builder)
        b.rng, b.q, b.ents = random.Random(i), "e0", ["e0"]
        b.attrs = {"a0": "bool", "a1": "bool", "a2": "int", "a3": "bool"}
        b.rules = [{"id": "r0", "if": [{"attr": "a0", "neg": False}], "then": {"attr": "a1", "neg": False}},
                   {"id": "r1", "if": [{"attr": "a2", "op": ">=", "value": 50}], "then": {"attr": "a3", "neg": False}}]
        b.exceptions, b.priorities = [], []
        b.facts = {("e0", "a0"): True, ("e0", "a2"): 7}
        b.walk(query, "True", 1)
        res = G.solve(b.theory(), query)
        assert (res["label"], res["depth"], b.facts[("e0", "a0")]) == ("True", 1, True)
        missing += ("e0", "a2") not in b.facts
    assert abs(missing / n - 0.25) < 0.06, missing / n
