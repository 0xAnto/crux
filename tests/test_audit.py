"""Tests for crux/gen/audit.py on tiny hand-built datasets."""
import copy

import numpy as np

from crux.gen import audit as A


def theory(n_ent=1, facts=(("a0", True),), rules=(((("a0", False),), "a1", False),)):
    """Make a theory. Each rule is (conditions, then_attr, then_neg). All facts are on e0."""
    attrs = {"a0", "a1"} | {a for _, a, _ in rules} | {c for cs, _, _ in rules for c, _ in cs}
    return {"entities": ["e%d" % i for i in range(n_ent)],
            "attributes": {a: "bool" for a in sorted(attrs)},
            "facts": [{"entity": "e0", "attr": a, "value": v} for a, v in facts],
            "rules": [{"id": "r%d" % i, "if": [{"attr": c, "neg": n} for c, n in cs], "then": {"attr": a, "neg": neg}}
                      for i, (cs, a, neg) in enumerate(rules)],
            "exceptions": [], "priorities": []}


def record(theories, t, q, unit, role, label, depth, *, block="train", flip=True, edit_type="fact_flip",
           path="/facts/0/value", options=("True", "False", "Unknown"), target=None, comp="c00"):
    h = A.canonical_hash(t)
    theories[h] = t
    tid, voc = {"train": ("t00", "v00"), "val": ("t09", "v08"), "test": ("t12", "v10")}[block]
    comp = {"train": comp, "val": "c08", "test": "c18"}[block]
    options = list(options)
    target = options.index(label) if target is None else target
    meta = {"template_id": tid, "template_group": "tg%d" % (int(tid[1:]) // 3), "composition_id": comp,
            "vocabulary_id": voc, "block": block, "depth": depth, "unit_depth": depth, "label": options[target],
            "unit_id": unit, "pair_id": f"{unit}-{tid}", "pair_role": role, "edit_type": edit_type,
            "edit_path": path, "label_flip": flip, "edit_relevant": True, "paraphrase_group": f"{unit}-{role}",
            "rule_types": [], "theory_hash": h, "query": q, "attempt": 0}
    return {"id": f"syn-{unit}-{tid}-{role}", "source": "synthetic",
            "task_type": "binary" if q["type"] == "binary" else "multi_choice",
            "state": f"state {unit} {role}", "instruction": f"rules {unit} {role}. Question: ?",
            "options": options, "target": target, "meta": meta}


def flip_unit(theories, k, block="train", n_ent=1, q=None):
    """A flip unit: a0 -> a1, and the edit sets a0 to false, so a1 becomes Unknown."""
    q = q or {"type": "binary", "entity": "e0", "attr": "a1", "neg": False}
    t = theory(n_ent)
    t2 = copy.deepcopy(t)
    t2["facts"][0]["value"] = False
    return [record(theories, t, q, f"u{k:06d}", "orig", "True", 1, block=block),
            record(theories, t2, q, f"u{k:06d}", "edit", "Unknown", 1, block=block)]


def names(out, status):
    return {c["name"] for c in out if c["status"] == status}


def test_attribute_shape_that_marks_stays_fails_flip_stay():
    theories, records = {}, []
    q = {"type": "binary", "entity": "e0", "attr": "a1", "neg": False}
    for k in range(400):
        block = "train" if k < 300 else "val"
        if k % 2:
            records += flip_unit(theories, k, block)
            continue
        # A stay: only these theories have an attribute (a2) that rules conclude only negatively.
        t = theory(facts=(("a0", True), ("a3", True)), rules=(((("a0", False),), "a1", False),
                                                              ((("a3", False),), "a2", True)))
        t2 = copy.deepcopy(t)
        del t2["facts"][1]
        u = f"u{k:06d}"
        records += [record(theories, t, q, u, "orig", "True", 1, block=block, flip=False,
                           edit_type="fact_remove", path="/facts/1"),
                    record(theories, t2, q, u, "edit", "True", 1, block=block, flip=False,
                           edit_type="fact_remove", path="/facts/1")]
    out = []
    A.check_flip_stay(out, A.build_units(A.build_items(records, theories)))
    assert names(out, "FAIL") == {f"flip_stay.{k}.{m}" for k in ("orig", "edit", "pair") for m in ("ood", "iid")}


def test_answer_always_highest_option_fails_mc_cues():
    theories, records = {}, []
    # a0 -> a1 -> a2 (true, height 2). a5 -> a3 and a5 -> a4 (unknown, height 1).
    t = theory(facts=(("a0", True),), rules=(((("a0", False),), "a1", False), ((("a1", False),), "a2", False),
                                             ((("a5", False),), "a3", False), ((("a5", False),), "a4", False)))
    orders = (["a2", "a3", "a4"], ["a3", "a2", "a4"], ["a3", "a4", "a2"])
    for k in range(120):
        cands = orders[k % 3]
        q = {"type": "multi_choice", "entity": "e0", "candidates": cands}
        opts = ["opt " + a for a in cands]
        records.append(record(theories, t, q, f"u{k:06d}", "orig", None, 2, block="train" if k < 90 else "val",
                              options=opts, target=cands.index("a2")))
    out = []
    A.check_mc_cues(out, A.build_items(records, theories))
    assert {"mc_cues.max_height", "mc_cues.unique_max_height"} <= names(out, "FAIL")


def test_clean_tiny_dataset_passes_invariants():
    theories, records = {}, []
    for k, block in enumerate(("train", "train", "val", "test")):
        records += flip_unit(theories, k, block, n_ent=k + 1)
    out = []
    A.check_invariants(out, records, theories, A.build_items(records, theories))
    assert names(out, "FAIL") == set()
    assert len(names(out, "PASS")) == 11


def test_duplicate_theory_and_query_fails_invariants():
    theories, records = {}, []
    records += flip_unit(theories, 0) + flip_unit(theories, 1)
    out = []
    A.check_invariants(out, records, theories, A.build_items(records, theories))
    assert "invariants.dup_theory_query" in names(out, "FAIL")


def test_flip_stay_without_signal_passes():
    """Labels that no feature predicts: the probe scores near the shuffled-label null, so the check passes even when
    the accuracy is below the majority rate."""
    theories, records = {}, []
    rng = __import__("random").Random(0)
    for k in range(400):
        recs = flip_unit(theories, k, "train" if k < 300 else "val", n_ent=rng.randint(1, 4))
        flip = rng.random() < 0.6
        for r in recs:
            r["meta"]["label_flip"] = flip
        records += recs
    out = []
    A.check_flip_stay(out, A.build_units(A.build_items(records, theories)))
    assert names(out, "FAIL") == set()


def test_weak_planted_cue_fails():
    """A weak cue (P(cue | flip) = 0.75, P(cue | stay) = 0.25) on noise features: an overfit model can score near
    the majority rate, but its accuracy is far above the shuffled-label null, so the check fails."""
    rng = np.random.default_rng(1)
    n = 1500
    y = rng.random(n) < 0.6
    cue = rng.random(n) < np.where(y, 0.75, 0.25)
    X = np.column_stack([rng.normal(size=(n, 20)), cue])
    blocks = np.where(np.arange(n) < 1100, "train", "val")
    for mode in ("ood", "iid"):
        out = []
        A.model_check(out, "cue", A.fit_eval(X, y, blocks, np.arange(n), mode), 0.05)
        assert out[0]["status"] == "FAIL", out


def test_lower_bound_uses_chance_agreement():
    """Unbalanced labels (70% "a") without signal. The model on the real labels predicts "a" and "b" in equal shares,
    and the shuffled-label model predicts the majority. The null is then about 0.2 above the accuracy, but the
    accuracy is at the chance agreement, so the check passes. A cue that is inverted on the test mask fails."""
    rng = np.random.default_rng(2)
    n = 2000
    y = np.where(rng.random(n) < 0.7, "a", "b")
    X = rng.normal(size=(n, 1))
    X[:, 0] = np.where(y == "a", 1.0, -1.0) * np.where(np.arange(n) < 1400, 1, 0) + rng.normal(size=n)
    blocks = np.where(np.arange(n) < 1400, "train", "val")
    real = lambda Xa, ya: np.mean((Xa[:, 0] > 0) == (ya == "a")) > 0.6  # the train labels follow feature 0

    def fit(Xa, ya, Xb):  # no signal on val: diverse predictions for real labels, the majority for shuffled ones
        return np.where(rng.random(len(Xb)) < 0.5, "a", "b") if real(Xa, ya) else np.full(len(Xb), "a")

    out = []
    A.model_check(out, "no_signal", A.fit_eval(X, y, blocks, np.arange(n), "ood", fit), 0.05)
    assert out[0]["status"] == "PASS" and out[0]["detail"]["null"] - out[0]["detail"]["acc"] > 0.1, out
    inverted = lambda Xa, ya, Xb: np.where(Xb[:, 0] > 0, "b", "a")  # val labels do not follow feature 0
    X[1400:, 0] = np.where(y[1400:] == "a", 1.0, -1.0) + 0.1 * rng.normal(size=n - 1400)
    out = []
    A.model_check(out, "inverted", A.fit_eval(X, y, blocks, np.arange(n), "ood", inverted), 0.05)
    assert out[0]["status"] == "FAIL", out


def test_null_within_strata_keeps_the_stratum_share():
    """Stratum 1 has 80% "u", stratum 0 has 1/3 of each label, and 10 noise features show the stratum. A null with
    all labels shuffled cannot learn the share of stratum 1, so acc - null is large. A null shuffled within each
    stratum learns it, so the check passes. A cue within the strata still fails."""
    rng = np.random.default_rng(3)
    n = 6000
    st = rng.random(n) < 0.5
    y = np.where(st, rng.choice(["t", "f", "u"], n, p=[0.1, 0.1, 0.8]), rng.choice(["t", "f", "u"], n))
    X = st[:, None] + 0.1 * rng.normal(size=(n, 10))
    blocks = np.where(np.arange(n) < 4000, "train", "val")
    vals = []
    for strata in (None, st):
        out = []
        A.model_check(out, "prior", A.fit_eval(X, y, blocks, np.arange(n), "ood", strata=strata), 0.03)
        vals.append(out[0]["value"])
    assert vals[0] > 0.03 and vals[1] <= 0.03, vals
    cue = (y == "u") & (rng.random(n) < 0.5)
    out = []
    A.model_check(out, "cue", A.fit_eval(np.column_stack([X, cue]), y, blocks, np.arange(n), "ood", strata=st), 0.03)
    assert out[0]["status"] == "FAIL", out


def test_qsplit_and_blind_split_the_facts_at_the_query_path():
    """a0 -> a1 (query a1) is the path; a2 -> a3 is a side rule. Only the facts on the path are drawn again."""
    t = theory(facts=(("a0", True), ("a2", False)), rules=(((("a0", False),), "a1", False), ((("a2", False),), "a3", False)))
    it = {"t": t, "q": {"type": "binary", "entity": "e0", "attr": "a1", "neg": False}, "h": A.heights(t)}
    assert A.qsplit(it) == [1, 0, 0, 0, 0, 1, 0, 0]
    rng = np.random.default_rng(0)
    for _ in range(20):
        t2, lab = A.blind(it, rng)
        assert {"entity": "e0", "attr": "a2", "value": False} in t2["facts"]
        assert lab == ("True" if {"entity": "e0", "attr": "a0", "value": True} in t2["facts"] else "Unknown")


def _path_items(planted):
    """P: a0 and a5 -> a1, N: a2 -> not a1 (query a1). With planted=False the facts on the path are random and the
    label follows from them. With planted=True each label has one fixed fact pattern (Unknown: all facts missing),
    a fact-count signal beyond the semantics."""
    rng = np.random.default_rng(3)
    pattern = {"True": (True, True, False), "False": (False, False, True), "Unknown": (None, None, None)}
    theories, records = {}, []
    q = {"type": "binary", "entity": "e0", "attr": "a1", "neg": False}
    rules = (((("a0", False), ("a5", False)), "a1", False), ((("a2", False),), "a1", True))
    k = 0
    while len(records) < 6000:
        lab = ("True", "False", "Unknown")[k % 3]
        vals = pattern[lab] if planted else tuple(None if u < 0.25 else bool(u < 0.625) for u in rng.random(3))
        t = theory(facts=[(a, v) for a, v in zip(("a0", "a5", "a2"), vals) if v is not None], rules=rules)
        if not planted:
            try:
                lab = A.label_depth(t, q, A.solve(t, "e0"), A.heights(t))[0]
            except ValueError:  # P and N both fire: not valid
                k += 1
                continue
        block = "train" if k % 4 else "val"
        records.append(record(theories, t, q, f"u{k:06d}", "orig", lab, 1, block=block))
        k += 1
    return A.build_items(records, theories)


def test_label_path_passes_random_facts_and_fails_a_planted_pattern():
    out = []
    A.check_label_path(out, _path_items(planted=False))
    assert names(out, "FAIL") == set(), out
    out = []
    A.check_label_path(out, _path_items(planted=True))
    assert names(out, "FAIL") == {"label_cue.path.orig.ood", "label_cue.path.orig.iid", "label_cue.ref.orig.ood",
                                  "label_cue.ref.orig.iid"}, out
