"""Step 4: evaluation metrics on schema records.

Input: a list of schema records and a parallel list of probability vectors (one value per option, sum 1).
The prediction is the argmax. For a `scoring` record, the correct option is the argmax of `target`.
Docs: docs/metrics.md. Each function with a dict result returns flat keys: a value, "<key>/n" for its size.
"""
import random
from collections import Counter, defaultdict

import numpy as np

IDS = ("template_id", "composition_id", "vocabulary_id")


def _gold(r):
    return int(np.argmax(r["target"])) if r["task_type"] == "scoring" else r["target"]


def _pred(r, p):
    p = np.asarray(p, dtype=float)
    if p.shape != (len(r["options"]),) or (p < 0).any() or abs(p.sum() - 1) > 1e-3:
        raise ValueError(f"record {r['id']}: the prediction must be {len(r['options'])} probabilities that sum to 1")
    return int(p.argmax())


def _hits(records, probs):
    if len(records) != len(probs):
        raise ValueError(f"{len(records)} records, but {len(probs)} predictions")
    return np.array([_pred(r, p) == _gold(r) for r, p in zip(records, probs)], dtype=bool)


def _put(out, name, flags):
    out[name] = float(np.mean(flags))
    out[name + "/n"] = len(flags)


def accuracy(records, probs):
    """Share of records where the predicted option is the correct option (higher is better)."""
    return float(_hits(records, probs).mean())


def macro_f1(records, probs):
    """Macro F1 of the option texts as classes (higher is better). Use it for binary records.
    The classes are all texts that occur as a correct or as a predicted option."""
    pairs = [(r["options"][_gold(r)], r["options"][_pred(r, p)]) for r, p in zip(records, probs)]
    f1 = []
    for c in sorted({t for pair in pairs for t in pair}):
        tp = sum(g == c and q == c for g, q in pairs)
        fp = sum(g != c and q == c for g, q in pairs)
        fn = sum(g == c and q != c for g, q in pairs)
        f1.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1))


def ece(records, probs, n_bins=15):
    """Expected calibration error on the maximum probability (lower is better).
    The bins have equal width on (0, 1]. The default is 15 bins, the value of Guo et al. (2017).
    The protocol does not fix the bin count. Report it with the result."""
    hit = _hits(records, probs)
    conf = np.array([np.max(p) for p in probs], dtype=float)
    bins = np.clip(np.ceil(conf * n_bins).astype(int) - 1, 0, n_bins - 1)
    return float(sum(abs(hit[bins == b].mean() - conf[bins == b].mean()) * (bins == b).mean()
                     for b in range(n_bins) if (bins == b).any()))


def paired_accuracy(records, probs):
    """Counterfactual paired accuracy (higher is better). A pair (`meta.pair_id`) counts as correct only if
    the orig and the edit record are both correct. Records without `pair_id` are ignored. Return {} if no record
    has one. Raise ValueError for an incomplete pair, because it means a split bug.
    Keys: paired_accuracy, then /flip, /stay, /relevant_stay (stay with `edit_relevant`), /unit_depth=<d>
    (`meta.unit_depth`; `depth` can differ between the two members), and
    /edit_type=<type>/orig=<label> (the orig label text for binary records, else the task type)."""
    pairs = defaultdict(dict)
    for r, h in zip(records, _hits(records, probs)):
        pid = r["meta"].get("pair_id")
        if pid is None:
            continue
        if r["meta"]["pair_role"] in pairs[pid]:
            raise ValueError(f"pair {pid}: two {r['meta']['pair_role']} records")
        pairs[pid][r["meta"]["pair_role"]] = (r, h)
    bad = [p for p, d in pairs.items() if set(d) != {"orig", "edit"}]
    if bad:
        raise ValueError(f"{len(bad)} incomplete pairs, for example {bad[:3]}")
    flags = defaultdict(list)
    for d in pairs.values():
        (o, ho), (_, he) = d["orig"], d["edit"]
        ok, m = bool(ho and he), o["meta"]
        lab = o["options"][_gold(o)] if o["task_type"] == "binary" else o["task_type"]
        flags["paired_accuracy"].append(ok)
        flags["paired_accuracy/" + ("flip" if m["label_flip"] else "stay")].append(ok)
        if m["edit_relevant"] and not m["label_flip"]:
            flags["paired_accuracy/relevant_stay"].append(ok)
        flags[f"paired_accuracy/unit_depth={m['unit_depth']}"].append(ok)
        flags[f"paired_accuracy/edit_type={m['edit_type']}/orig={lab}"].append(ok)
    out = {}
    for k in sorted(flags, key=lambda k: (k.count("/"), k)):
        _put(out, k, flags[k])
    return out


def paraphrase_consistency(records, probs):
    """Over the groups of `meta.paraphrase_group` with 2 or more records (higher is better).
    Keys: paraphrase_consistency (share of groups where all members predict the same option text),
    paraphrase_consistent_correct (the same, and the option is correct), paraphrase_groups (number of groups).
    A group with one record is skipped. Return {} if no group is left."""
    groups = defaultdict(list)
    for r, p in zip(records, probs):
        g = r["meta"].get("paraphrase_group")
        if g is not None:
            groups[g].append((r["options"][_pred(r, p)], r["options"][_gold(r)]))
    groups = [g for g in groups.values() if len(g) > 1]
    if not groups:
        return {}
    same = [len({q for q, _ in g}) == 1 for g in groups]
    both = [s and all(q == t for q, t in g) for s, g in zip(same, groups)]
    return {"paraphrase_consistency": float(np.mean(same)), "paraphrase_consistent_correct": float(np.mean(both)),
            "paraphrase_groups": len(groups)}


def permute_options(record, seed):
    """Return a copy of the record with permuted options. The permutation is never the identity.
    It depends only on `seed` and the record id. `target` follows the options (for `scoring`, the scores are
    permuted). `meta.option_perm` holds the permutation: new options[i] = old options[option_perm[i]].
    The result is for evaluation only: a binary record with permuted options breaks the canonical order."""
    n = len(record["options"])
    if n < 2:
        raise ValueError(f"record {record['id']}: a permutation needs 2 or more options")
    rng, perm = random.Random(f"{seed}/{record['id']}"), list(range(n))
    while perm == list(range(n)):
        rng.shuffle(perm)
    t = record["target"]
    target = [t[i] for i in perm] if record["task_type"] == "scoring" else perm.index(t)
    return {**record, "options": [record["options"][i] for i in perm], "target": target,
            "meta": {**record["meta"], "option_perm": perm}}


def order_invariance(records_by_perm, probs_by_perm):
    """Share of records whose predicted option text is the same under all permutations (higher is better).
    `records_by_perm[k]` holds the records of permutation k (include the identity). `probs_by_perm[k]` holds the
    predictions for them. All permutations list the records in the same order."""
    if len(records_by_perm) < 2:
        raise ValueError("order invariance needs 2 or more permutations")
    ids = [[r["id"] for r in rs] for rs in records_by_perm]
    if any(i != ids[0] for i in ids):
        raise ValueError("the permutations must hold the same record ids in the same order")
    if len(probs_by_perm) != len(ids) or any(len(p) != len(i) for p, i in zip(probs_by_perm, ids)):
        raise ValueError("each permutation needs one prediction for each record")
    texts = [[r["options"][_pred(r, p)] for r, p in zip(rs, ps)] for rs, ps in zip(records_by_perm, probs_by_perm)]
    return float(np.mean([len(set(t)) == 1 for t in zip(*texts)]))


def seen_ids(train):
    """The sets of template, composition, and vocabulary ids in the train records."""
    return {k: {r["meta"][k] for r in train} for k in IDS}


def _leaked(r, seen):
    return [k for k in IDS if r["meta"][k] in seen[k]]


def unseen_rule_accuracy(records, probs, seen, drop_seen=False):
    """Accuracy on records whose template, composition, and vocabulary ids are all absent from `seen`
    (the result of `seen_ids`) (higher is better). A record with a seen id means a split bug: raise ValueError,
    or drop the record if `drop_seen`. Keys: unseen_rule_accuracy, then /features=<k> (number of set bits of
    `composition_id`) and /depth=<d>. Return {} if no record is left."""
    hit, keep = _hits(records, probs), []
    for i, r in enumerate(records):
        leaked = _leaked(r, seen)
        if leaked and not drop_seen:
            raise ValueError(f"record {r['id']} has train ids: {leaked}")
        if not leaked:
            keep.append(i)
    flags = defaultdict(list)
    for i in keep:
        m = records[i]["meta"]
        for k in ("unseen_rule_accuracy", f"unseen_rule_accuracy/features={bin(int(m['composition_id'][1:])).count('1')}",
                  f"unseen_rule_accuracy/depth={m['depth']}"):
            flags[k].append(hit[i])
    out = {}
    for k in sorted(flags, key=lambda k: (k.count("/"), k)):
        _put(out, k, flags[k])
    return out


def _label(r):
    return r["options"][_gold(r)] if r["task_type"] == "binary" else _gold(r)


def _top(records):
    c = defaultdict(Counter)
    for r in records:
        c[r["task_type"]][_label(r)] += 1
    return {t: k.most_common(1)[0][0] for t, k in c.items()}


def majority_baseline(records, by=None, ref=None):
    """Accuracy of the constant prediction of the most frequent label (higher is better).
    The label is the option text (binary) or the target index (other types), counted per task type.
    A tie goes to the label that occurs first. The majority comes from `ref` (for example the train records),
    else from `records`. With `by` (a meta key), each group of `records` uses the majority of the `ref` records
    of the same group. A group that `ref` does not have uses the majority of all `ref`.
    Keys: majority_baseline, or majority_baseline/<by>=<group> for each group."""
    ref = records if ref is None else ref
    overall = _top(ref)
    if by is None:
        return {"majority_baseline": float(np.mean([_label(r) == overall.get(r["task_type"]) for r in records]))}
    groups, ref_groups = defaultdict(list), defaultdict(list)
    for r in records:
        groups[r["meta"][by]].append(r)
    for r in ref:
        ref_groups[r["meta"][by]].append(r)
    out = {}
    for g in sorted(groups):
        top = {**overall, **_top(ref_groups.get(g, []))}
        out[f"majority_baseline/{by}={g}"] = float(np.mean([_label(r) == top.get(r["task_type"]) for r in groups[g]]))
    return out


def evaluate(records, probs, train=None, perms=None):
    """All metrics that apply, as a flat dict. A metric is skipped if the meta fields it needs are absent.
    `train`: the train records. They give the seen ids (unseen-rule accuracy) and the majority label.
    `perms`: (records_by_perm, probs_by_perm) for option-order invariance.
    If the records come from more than one `source` or `meta.subset`, there is no pooled value: each key has
    the prefix `<source>/<subset>/` (for example `proofwriter/depth5/accuracy`), because a pooled accuracy over
    mixed sources is not a result.
    Unseen-rule accuracy uses the records with all of `template_id`, `composition_id`, `vocabulary_id`. It drops
    the records with a train id (dev_id has only such records) and reports their number as
    `unseen_rule_accuracy/seen_dropped`. Its `/n` is the number of records it scored."""
    if len(records) != len(probs):
        raise ValueError(f"{len(records)} records, but {len(probs)} predictions")
    groups = defaultdict(list)
    for i, r in enumerate(records):
        groups[r["source"] + ("/" + r["meta"]["subset"] if "subset" in r["meta"] else "")].append(i)
    if len(groups) > 1:
        return {f"{g}/{k}": v for g, ix in sorted(groups.items()) for k, v in evaluate(
            [records[i] for i in ix], [probs[i] for i in ix], train,
            perms and ([[rs[i] for i in ix] for rs in perms[0]], [[ps[i] for i in ix] for ps in perms[1]])).items()}
    out = {"n": len(records), "accuracy": accuracy(records, probs)}
    binary = [i for i, r in enumerate(records) if r["task_type"] == "binary"]
    if binary:
        out["macro_f1"] = macro_f1([records[i] for i in binary], [probs[i] for i in binary])
    out["ece"] = ece(records, probs)
    out.update(paired_accuracy(records, probs))
    out.update(paraphrase_consistency(records, probs))
    if perms is not None:
        out["option_order_invariance"] = order_invariance(*perms)
    have = [i for i, r in enumerate(records) if all(k in r["meta"] for k in IDS)]
    if train is not None and have:
        seen = seen_ids(train)
        out.update(unseen_rule_accuracy([records[i] for i in have], [probs[i] for i in have], seen, drop_seen=True))
        out["unseen_rule_accuracy/seen_dropped"] = sum(bool(_leaked(records[i], seen)) for i in have)
    for by in (None, "composition_id"):
        if by is None or all(by in r["meta"] for r in records):
            out.update(majority_baseline(records, by, train))
    return out
