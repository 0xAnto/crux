import pytest

from crux import metrics as M

TFU = ["True", "False", "Unknown"]


def rec(i, target, options=TFU, task="binary", **meta):
    return {"id": i, "source": "t", "task_type": task, "state": "s", "instruction": "q", "options": list(options),
            "target": target, "meta": meta}


def hot(k, n=3, conf=1.0):
    return [conf if j == k else (1 - conf) / (n - 1) for j in range(n)]


def pair(pid, t, ok_orig, ok_edit, **meta):
    """Two records with label index t. A wrong member predicts the next option."""
    return [rec(f"{pid}-o", t, pair_id=pid, pair_role="orig", **meta), rec(f"{pid}-e", t, pair_id=pid, pair_role="edit", **meta)], \
           [hot(t if ok_orig else (t + 1) % 3), hot(t if ok_edit else (t + 1) % 3)]


def test_paired_accuracy_and_breakdown():
    parts = [pair("a", 0, 1, 1, label_flip=True, edit_relevant=True, edit_type="fact_flip", unit_depth=1),
             pair("b", 0, 1, 0, label_flip=True, edit_relevant=True, edit_type="fact_flip", unit_depth=1),
             pair("c", 2, 1, 1, label_flip=False, edit_relevant=True, edit_type="rule_remove", unit_depth=2),
             pair("d", 2, 0, 1, label_flip=False, edit_relevant=False, edit_type="rule_remove", unit_depth=2)]
    recs, probs = [x for p in parts for x in p[0]], [x for p in parts for x in p[1]]
    out = M.paired_accuracy(recs, probs)
    assert out["paired_accuracy"] == 0.5 and out["paired_accuracy/n"] == 4
    assert out["paired_accuracy/flip"] == 0.5 and out["paired_accuracy/stay"] == 0.5
    assert (out["paired_accuracy/relevant_stay"], out["paired_accuracy/relevant_stay/n"]) == (1.0, 1)
    assert out["paired_accuracy/edit_type=fact_flip/orig=True"] == 0.5
    assert (out["paired_accuracy/unit_depth=1"], out["paired_accuracy/unit_depth=2"]) == (0.5, 0.5)
    assert out["paired_accuracy/unit_depth=1/n"] == 2
    assert out["paired_accuracy/edit_type=rule_remove/orig=Unknown"] == 0.5
    assert M.accuracy(recs, probs) == 0.75  # a pair with one wrong member counts 0, a record does not
    with pytest.raises(ValueError, match="incomplete"):
        M.paired_accuracy(recs[:-1], probs[:-1])
    assert M.paired_accuracy([rec("x", 0)], [hot(0)]) == {}


def test_ece_hand_value():
    recs = [rec(str(i), 0, TFU[:2]) for i in range(4)]
    probs = [[.85, .15], [.15, .85], [.65, .35], [.65, .35]]
    # 10 bins. Confidence .85: records 0 and 1 (hits 1, 0), |0.5 - 0.85| = 0.35. Confidence .65: records 2 and 3
    # (hits 1, 1), |1 - 0.65| = 0.35. Each bin has weight 0.5.
    assert M.ece(recs, probs, n_bins=10) == pytest.approx(0.35)
    assert M.ece(recs, [[1., 0.]] * 4) == 0.0


def test_paraphrase_consistency_skips_singleton():
    g = lambda grp, t: rec(f"{grp}{t}", 0, paraphrase_group=grp)
    recs = [g("g1", 0), g("g1", 1), g("g1", 2), g("g2", 0), g("g2", 1), g("g3", 0), g("g4", 0), g("g4", 1)]
    probs = [hot(0)] * 3 + [hot(0), hot(1)] + [hot(1)] + [hot(2)] * 2  # g3 singleton is wrong, g4 same but wrong
    out = M.paraphrase_consistency(recs, probs)
    assert out == {"paraphrase_consistency": pytest.approx(2 / 3), "paraphrase_consistent_correct": pytest.approx(1 / 3),
                   "paraphrase_groups": 3}
    assert M.paraphrase_consistency(recs[5:6], probs[5:6]) == {}


def test_permute_options_keeps_target_text_and_is_deterministic():
    r = rec("m", 2, ["a", "b", "c", "d"], task="multi_choice")
    p = M.permute_options(r, 7)
    assert p == M.permute_options(r, 7) and p["options"] != r["options"]
    assert p["options"][p["target"]] == "c" and sorted(p["options"]) == r["options"]
    assert [r["options"][i] for i in p["meta"]["option_perm"]] == p["options"] and "option_perm" not in r["meta"]
    s = rec("s", [0.1, 0.9, 0.5], ["a", "b", "c"], task="scoring")
    q = M.permute_options(s, 1)
    assert q["options"][q["target"].index(0.9)] == "b" and sorted(q["target"]) == [0.1, 0.5, 0.9]
    assert M.permute_options(rec("two", 0), 0)["options"] != TFU
    for n in (0, 1):
        with pytest.raises(ValueError, match="2 or more options"):
            M.permute_options(rec("few", 0, TFU[:n]), 0)


def test_permute_options_target_follows_the_text_for_every_seed_and_target():
    for t in range(4):
        r = rec("m", t, ["a", "b", "c", "d"], task="multi_choice")
        for seed in range(30):
            p = M.permute_options(r, seed)
            assert p["options"][p["target"]] == r["options"][t] and p["meta"]["option_perm"][p["target"]] == t


def test_order_invariance():
    recs = [rec("r1", 0), rec("r2", 1)]
    perm = [M.permute_options(r, 3) for r in recs]
    at = lambda r, text: hot(r["options"].index(text))
    base = [at(recs[0], "True"), at(recs[1], "False")]
    moved = [at(perm[0], "True"), at(perm[1], "Unknown")]  # the second prediction changes with the order
    assert M.order_invariance([recs, perm], [base, moved]) == 0.5
    assert M.order_invariance([recs, perm], [base, [at(perm[0], "True"), at(perm[1], "False")]]) == 1.0
    with pytest.raises(ValueError, match="one prediction"):
        M.order_invariance([recs, perm], [base, moved[:1]])
    with pytest.raises(ValueError, match="one prediction"):
        M.order_invariance([recs, perm], [base])
    with pytest.raises(ValueError, match="same record ids"):
        M.order_invariance([recs, perm[:1]], [base, moved[:1]])


def test_unseen_rule_accuracy_and_leak():
    seen = M.seen_ids([rec("t", 0, template_id="t01", composition_id="c03", vocabulary_id="v07", depth=1)])
    ids = lambda i, c, d: dict(template_id="t09", composition_id=c, vocabulary_id="v08", depth=d)
    recs = [rec("a", 0, **ids(0, "c05", 1)), rec("b", 0, **ids(0, "c05", 1)), rec("c", 0, **ids(0, "c23", 3))]
    probs = [hot(0), hot(1), hot(0)]
    out = M.unseen_rule_accuracy(recs, probs, seen)
    assert out["unseen_rule_accuracy"] == pytest.approx(2 / 3)
    assert out["unseen_rule_accuracy/features=2"] == 0.5 and out["unseen_rule_accuracy/features=4"] == 1.0
    assert out["unseen_rule_accuracy/depth=1"] == 0.5 and out["unseen_rule_accuracy/depth=3"] == 1.0
    leak = rec("x", 0, template_id="t01", composition_id="c05", vocabulary_id="v08", depth=1)
    with pytest.raises(ValueError, match="train ids"):
        M.unseen_rule_accuracy(recs + [leak], probs + [hot(0)], seen)
    assert M.unseen_rule_accuracy(recs + [leak], probs + [hot(1)], seen, drop_seen=True)["unseen_rule_accuracy/n"] == 3


def test_macro_f1_hand_value():
    gold, pred = [0, 0, 1, 2], [0, 1, 1, 1]  # True: F1 2/3, False: 2/4, Unknown: 0
    recs = [rec(str(i), g) for i, g in enumerate(gold)]
    assert M.macro_f1(recs, [hot(p) for p in pred]) == pytest.approx((2 / 3 + 0.5 + 0) / 3)


def test_majority_baseline_uses_reference():
    t, u = lambda i, g: rec(i, 0, composition_id=g), lambda i, g: rec(i, 2, composition_id=g)
    ref = [u("r1", "c00"), u("r2", "c00"), t("r3", "c00"), t("r4", "c01"), t("r5", "c01"), t("r6", "c01")]  # c00: Unknown, c01: True
    recs = [t("a", "c00"), t("b", "c00"), u("c", "c00"), u("d", "c02"), u("e", "c02"), t("f", "c02")]  # own majority: c00 True, c02 Unknown
    assert M.majority_baseline(recs[3:], ref=ref)["majority_baseline"] == 1 / 3  # the reference majority is True (4 of 6)
    assert M.majority_baseline(recs[3:])["majority_baseline"] == 2 / 3  # no reference: the own majority
    out = M.majority_baseline(recs, by="composition_id", ref=ref)
    # c00 uses the c00 majority of ref (Unknown). c02 is not in ref: it uses the majority of all ref (True).
    assert out == {"majority_baseline/composition_id=c00": 1 / 3, "majority_baseline/composition_id=c02": 1 / 3}
    assert M.majority_baseline(recs, by="composition_id")["majority_baseline/composition_id=c00"] == 2 / 3
    mc = [rec("m", 1, ["a", "b"], task="multi_choice")]
    assert M.majority_baseline(mc, ref=mc + [rec("n", 0, ["a", "b"], task="multi_choice")] * 2)["majority_baseline"] == 0.0


def test_evaluate_skips_absent_meta():
    recs = [rec("a", 0), rec("b", 1)]
    out = M.evaluate(recs, [hot(0), hot(0)], train=recs)
    assert set(out) == {"n", "accuracy", "macro_f1", "ece", "majority_baseline"} and out["accuracy"] == 0.5


def test_evaluate_reports_each_source_and_subset_and_no_pooled_value():
    recs = [rec("a", 0, subset="s1"), rec("b", 1, subset="s1"), rec("c", 0, subset="s2"), {**rec("d", 0), "source": "u"}]
    probs = [hot(0), hot(0), hot(0), hot(1)]
    out = M.evaluate(recs, probs)
    assert (out["t/s1/accuracy"], out["t/s2/accuracy"], out["u/accuracy"]) == (0.5, 1.0, 0.0)
    assert (out["t/s1/n"], out["t/s2/n"], out["u/n"]) == (2, 1, 1)
    assert "accuracy" not in out and "n" not in out and "majority_baseline" not in out
    perms = ([recs, [M.permute_options(r, 1) for r in recs]], [probs, probs])
    assert "t/s1/option_order_invariance" in M.evaluate(recs, probs, perms=perms)


def test_evaluate_unseen_rule_accuracy_uses_the_records_with_ids_and_survives_seen_ones():
    ids = lambda i, t: rec(i, 0, template_id=t, composition_id="c05", vocabulary_id="v08", depth=1)
    train = [rec("t", 0, template_id="t01", composition_id="c03", vocabulary_id="v07", depth=1)]
    recs = [ids("a", "t09"), ids("b", "t09"), ids("c", "t01"), rec("no-ids", 0)]  # c has a train template, no-ids has no ids
    out = M.evaluate(recs, [hot(0), hot(1), hot(0), hot(0)], train=train)
    assert (out["unseen_rule_accuracy"], out["unseen_rule_accuracy/n"], out["unseen_rule_accuracy/seen_dropped"]) == (0.5, 2, 1)
    dev = M.evaluate([train[0]], [hot(0)], train=train)  # an in-distribution set: every record is seen
    assert "unseen_rule_accuracy" not in dev and dev["unseen_rule_accuracy/seen_dropped"] == 1
    assert not any(k.startswith("majority_baseline/block") for k in M.evaluate(recs, [hot(0)] * 4, train=train))
