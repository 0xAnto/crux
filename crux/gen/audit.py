"""Audit the Step 3 synthetic dataset for label shortcuts and broken invariants.

Usage: python -m crux.gen.audit <data_dir>

The audit reads all.jsonl, theories.jsonl, and report.json in <data_dir>. It uses only the
data format and the semantics of docs/generator_spec.md. It does not import the generator,
the engine, or the templates, so that it stays independent of the code that it audits.
All thresholds are fixed in this file. The trained models use a fixed seed. Each shortcut probe
runs on two splits: "ood" (train block -> val block) and "iid" (two halves of the train block,
split by unit). A probe passes if its accuracy is not above the accuracy of the same model trained
on shuffled labels (the null) or the majority rate, and not below the chance agreement of its own
predictions (an inverted cue).

The audit prints one line for each check, writes audit.json in <data_dir>, and exits with
code 1 if a check fails.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SEED = 0
HELD_COMPS = {"val": {"c08", "c05", "c11", "c23"}, "test": {"c18", "c28", "c31"}}
PART = {  # Spec section 4.2: the permitted IDs and record depths of each block.
    "train": {"comp": {"c%02d" % i for i in range(32)} - HELD_COMPS["val"] - HELD_COMPS["test"],
              "tg": {"tg0", "tg1", "tg2"}, "voc": {"v%02d" % i for i in range(8)}, "depth": range(4)},
    "val": {"comp": HELD_COMPS["val"], "tg": {"tg3"}, "voc": {"v08", "v09"}, "depth": range(6)},
    "test": {"comp": HELD_COMPS["test"], "tg": {"tg4"}, "voc": {"v10", "v11"}, "depth": range(6)},
}
OPS = {">=": lambda v, k: v >= k, ">": lambda v, k: v > k, "<=": lambda v, k: v <= k,
       "<": lambda v, k: v < k, "==": lambda v, k: v == k}


# ---------------------------------------------------------------- evaluator (spec sections 2 and 3)

def solve(t, e):
    """Return {attr: (value, depth)} for entity e. Raise ValueError on an unresolved conflict."""
    out = {f["attr"]: (f["value"], 0) for f in t["facts"] if f["entity"] == e}
    by, blocks = defaultdict(list), defaultdict(list)
    for r in t["rules"]:
        by[r["then"]["attr"]].append(r)
    for x in t["exceptions"]:
        blocks[x["blocks"]].append(x)
    beats = {(p["higher"], p["lower"]) for p in t["priorities"]}

    def get(a):
        if a not in out and a in by:
            out[a] = derive(a)
        return out.get(a, (None, None))

    def holds(c):
        v = get(c["attr"])[0]
        return v is not None and (OPS[c["op"]](v, c["value"]) if "op" in c else v == (not c["neg"]))

    def derive(a):
        fire = [r for r in by[a] if all(map(holds, r["if"]))
                and not any(all(map(holds, x["if"])) for x in blocks[r["id"]])]
        pos = [r for r in fire if not r["then"]["neg"]]
        neg = [r for r in fire if r["then"]["neg"]]
        if pos and neg:
            pos, neg = ([r for r in pos if all((r["id"], s["id"]) in beats for s in neg)],
                        [r for r in neg if all((r["id"], s["id"]) in beats for s in pos)])
            if not pos and not neg:
                raise ValueError("conflict on " + a)
        win = pos or neg
        if not win:
            return (None, None)
        d = min(1 + max(0 if "op" in c else get(c["attr"])[1] for c in r["if"]) for r in win)
        return (bool(pos), d)

    for a in t["attributes"]:
        get(a)
    return out


def heights(t):
    """Return the attribute height h(a) of spec section 3.2 for each attribute."""
    by, h = defaultdict(list), {}
    for r in t["rules"]:
        by[r["then"]["attr"]].append(r)

    def f(a):
        if a not in h:
            h[a] = max((1 + max(f(c["attr"]) for c in r["if"]) for r in by[a]), default=0)
        return h[a]
    return {a: f(a) for a in t["attributes"]}


def label_depth(t, q, s, h):
    """Return (label, depth) of query q from the model s. The label is None if the query is invalid."""
    if q["type"] == "binary":
        v, d = s.get(q["attr"], (None, None))
        if v is None:
            return "Unknown", h[q["attr"]]
        return ("True" if v == (not q["neg"]) else "False"), d
    true = [c for c in q["candidates"] if s.get(c, (None,))[0] is True]
    return (true[0], s[true[0]][1]) if len(true) == 1 else (None, None)


# ---------------------------------------------------------------- loading

def load(data_dir):
    d = Path(data_dir)
    records = [json.loads(line) for line in open(d / "all.jsonl", encoding="utf-8")]
    theories = {}
    for line in open(d / "theories.jsonl", encoding="utf-8"):
        o = json.loads(line)
        theories[o["theory_hash"]] = o["theory"]
    report = json.load(open(d / "report.json")) if (d / "report.json").exists() else {}
    return records, theories, report


def canonical_hash(t):
    s = json.dumps(t, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def build_items(records, theories):
    """Return one item for each paraphrase group (the first record), with its theory and model."""
    items = {}
    for r in records:
        g = r["meta"]["paraphrase_group"]
        if g in items or r["meta"]["theory_hash"] not in theories:
            continue
        t, q = theories[r["meta"]["theory_hash"]], r["meta"]["query"]
        h = heights(t)
        try:
            s = solve(t, q["entity"])
            lab, dep = label_depth(t, q, s, h)
        except (ValueError, KeyError, RecursionError):
            s, lab, dep = None, None, None
        items[g] = {"rec": r, "m": r["meta"], "t": t, "q": q, "h": h, "s": s, "label": lab, "depth": dep}
    return items


def build_units(items):
    units = defaultdict(dict)
    for it in items.values():
        units[it["m"]["unit_id"]][it["m"]["pair_role"]] = it
    return {u: v for u, v in units.items() if "orig" in v and "edit" in v}


# ---------------------------------------------------------------- shallow features (no fact lookup, no solving)

def qattrs(q):
    return [q["attr"]] if q["type"] == "binary" else q["candidates"]


def shape(it):
    """Return shallow structural counts of the theory and the query of an item."""
    t, q, h = it["t"], it["q"], it["h"]
    concl, reads, lits = defaultdict(list), Counter(), Counter()
    for r in t["rules"]:
        concl[r["then"]["attr"]].append(r["then"]["neg"])
    for x in t["rules"] + t["exceptions"]:
        for c in x["if"]:
            reads[c["attr"]] += 1
            lits["thr" if "op" in c else ("neg" if c["neg"] else "pos")] += 1
            if "op" not in c and not c["neg"]:
                lits["pos:" + c["attr"]] += 1
    mask = [a for a, v in concl.items() if all(v) and reads[a] == 1 and lits["pos:" + a] == 1]
    fact_attrs = {f["attr"] for f in t["facts"]}
    base = [a for a in t["attributes"] if a not in concl]
    qa = qattrs(q)
    qr = [r for r in t["rules"] if r["then"]["attr"] in qa]
    qids = {r["id"] for r in qr}
    nconds = [len(r["if"]) for r in t["rules"]] or [0]
    qf = {f["attr"]: f["value"] for f in t["facts"] if f["entity"] == q["entity"]}  # the facts of the queried entity
    qb = [a for a in base if t["attributes"][a] == "bool"]
    return [len(t["entities"]), len(t["facts"]), sum(f["value"] is True for f in t["facts"]),
            sum(f["value"] is False for f in t["facts"]), len(t["rules"]), len(t["exceptions"]),
            len(t["priorities"]), len(t["attributes"]), len(concl), sum(all(v) for v in concl.values()),
            sum(not any(v) for v in concl.values()), sum(n == 1 for n in reads.values()), len(mask),
            sum(reads[a] == 1 and a not in fact_attrs for a in base), lits["pos"], lits["neg"], lits["thr"],
            max(h.values(), default=0), sum(h[a] for a in qa), max(h[a] for a in qa),
            sum(not r["then"]["neg"] for r in qr), sum(r["then"]["neg"] for r in qr),
            sum(x["blocks"] in qids for x in t["exceptions"]), sum(p["higher"] in qids for p in t["priorities"]),
            sum(p["lower"] in qids for p in t["priorities"]), int(q.get("neg", -1)), len(qa),
            max(nconds), float(np.mean(nconds)), sum(qf.get(a) is True for a in qb), sum(qf.get(a) is False for a in qb),
            sum(a not in qf for a in qb), sum(t["attributes"][a] == "int" for a in qf)]


def option_shape(it, a):
    """Return shallow counts for one multi-choice option (attribute a)."""
    t, h = it["t"], it["h"]
    rs = [r for r in t["rules"] if r["then"]["attr"] == a]
    ids = {r["id"] for r in rs}
    reads = sum(c["attr"] == a for x in t["rules"] + t["exceptions"] for c in x["if"])
    return [h[a], sum(not r["then"]["neg"] for r in rs), sum(r["then"]["neg"] for r in rs),
            sum(x["blocks"] in ids for x in t["exceptions"]), sum(p["higher"] in ids for p in t["priorities"]),
            sum(p["lower"] in ids for p in t["priorities"]), reads, int(any(f["attr"] == a for f in t["facts"])),
            sum(len(r["if"]) for r in rs), len(it["q"]["candidates"])]


def query_path(t, q):
    """The attributes on the query path: the query attributes and, transitively, the attributes that the rules and
    exceptions toward them read."""
    by, exc = defaultdict(list), defaultdict(list)
    for r in t["rules"]:
        by[r["then"]["attr"]].append(r)
    for x in t["exceptions"]:
        exc[x["blocks"]].append(x)
    path, todo = set(), list(qattrs(q))
    while todo:
        a = todo.pop()
        if a not in path:
            path.add(a)
            todo += [c["attr"] for r in by.get(a, ()) for c in r["if"] + [c for x in exc.get(r["id"], ()) for c in x["if"]]]
    return path


def qsplit(it):
    """The facts of the queried entity on the query path, then off the path: true, false, and missing bool base
    facts, and int facts."""
    t, q = it["t"], it["q"]
    path, heads = query_path(t, q), {r["then"]["attr"] for r in t["rules"]}
    qf = {f["attr"]: f["value"] for f in t["facts"] if f["entity"] == q["entity"]}
    out = []
    for on in (True, False):
        b = [a for a, typ in t["attributes"].items() if typ == "bool" and a not in heads and (a in path) == on]
        out += [sum(qf.get(a) is True for a in b), sum(qf.get(a) is False for a in b), sum(a not in qf for a in b),
                sum(t["attributes"][a] == "int" and (a in path) == on for a in qf)]
    return out


def blind(it, rng):
    """The theory of item it with the facts of the queried entity on the query path drawn again at random (bool: 3/8
    true, 3/8 false, 1/4 missing; int: 3/4 a value 0..99, 1/4 missing), and its label (None if not valid)."""
    t, q, e = it["t"], it["q"], it["q"]["entity"]
    path = query_path(t, q) - {r["then"]["attr"] for r in t["rules"]}
    facts = [f for f in t["facts"] if not (f["entity"] == e and f["attr"] in path)]
    for a in sorted(path, key=lambda x: (len(x), x)):
        u = rng.random()
        if u < 0.75:
            v = bool(u < 0.375) if t["attributes"][a] == "bool" else int(rng.integers(0, 100))
            facts.append({"entity": e, "attr": a, "value": v})
    t2 = {**t, "facts": facts}
    try:
        return t2, label_depth(t2, q, solve(t2, e), it["h"])[0]
    except (ValueError, KeyError, RecursionError):
        return t2, None


def _balanced(rows, cap=None, seed=SEED):
    """Keep the same number of rows of each label in each (block, composition): the smallest count, at most cap[key]
    if cap is given. A row is (features, label, block, group, composition). Return (rows, {key: count})."""
    rng = np.random.default_rng(seed)
    groups = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(rows):
        groups[(r[2], r[4])][r[1]].append(i)
    keep, counts = [], {}
    for key in sorted(groups):
        g = groups[key]
        if len(g) == 3:
            counts[key] = n = min([len(v) for v in g.values()] + ([cap.get(key, 0)] if cap is not None else []))
            keep += [i for lab in sorted(g) for i in rng.choice(g[lab], n, replace=False)]
    return [rows[i] for i in sorted(keep)], counts


def gbm():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=200, random_state=SEED)


def _split(blocks, groups, mode):
    """Train and test masks. "ood": train block -> val block. "iid": two halves of the train block, split by group
    (unit), so that no unit is in both halves."""
    b = np.array(blocks)
    if mode == "ood":
        return b == "train", b == "val"
    half = np.array([int(hashlib.sha256(str(g).encode()).hexdigest(), 16) % 2 for g in groups])
    return (b == "train") & (half == 0), (b == "train") & (half == 1)


def fit_eval(X, y, blocks, groups=None, mode="ood", fit=None, strata=None):
    """Return (accuracy, majority rate, null accuracy, chance agreement) on the test mask, or None without data.
    The null accuracy is the mean accuracy of the same model trained on shuffled train labels (5 shuffles). A model
    without signal scores near the null, and the null is below the majority rate when the model overfits noise.
    With `strata`, the labels are shuffled only within each stratum, so the null keeps the label share of each
    stratum.
    The chance agreement is sum over labels of P(prediction = l) * P(label = l): the accuracy of predictions with the
    same shares that are independent of the labels."""
    fit = fit or (lambda Xa, ya, Xb: gbm().fit(Xa, ya).predict(Xb))
    y = np.array(y)
    tr, te = _split(blocks, groups if groups is not None else range(len(y)), mode)
    if te.sum() == 0 or len(set(y[tr])) < 2:
        return None
    Xtr = X[tr] if hasattr(X, "shape") else np.array(X, dtype=float)[tr]
    Xte = X[te] if hasattr(X, "shape") else np.array(X, dtype=float)[te]
    pred = fit(Xtr, y[tr], Xte)
    acc = float(np.mean(pred == y[te]))
    chance = float(sum(np.mean(pred == v) * np.mean(y[te] == v) for v in set(y[te])))
    rng = np.random.default_rng(SEED)
    st = np.array(strata)[tr] if strata is not None else np.zeros(tr.sum())
    ytr = y[tr]

    def shuffled():
        out = ytr.copy()
        for v in set(st):
            i = np.flatnonzero(st == v)
            out[i] = ytr[rng.permutation(i)]
        return out
    null = float(np.mean([np.mean(fit(Xtr, shuffled(), Xte) == y[te]) for _ in range(5)]))
    return acc, Counter(y[te]).most_common(1)[0][1] / te.sum(), null, chance


# ---------------------------------------------------------------- checks

def check(out, name, value, ok, threshold, detail=None):
    out.append({"name": name, "value": value, "threshold": threshold,
                "status": "WARN" if ok is None else ("PASS" if ok else "FAIL"), "detail": detail})


def model_check(out, name, res, tol):
    """Pass if acc - null <= tol, acc - majority <= tol, and chance - acc <= tol. The null is the accuracy without
    signal, so acc - null (the check value) shows the signal even when an overfit model scores below the majority.
    The lower bound uses the chance agreement and not the null: with unbalanced labels, a shuffled-label model
    predicts the majority label more often than a model trained on the real labels, so a model without signal can
    score more than tol below the null (label_cue.bow.ood: acc 0.321, null 0.367, chance 0.327). A model below its
    chance agreement has a cue that is inverted on the test mask."""
    th = f"acc-null<={tol}, acc-maj<={tol}, chance-acc<={tol}"
    if res is None:
        return check(out, name, None, None, th, "no train or test data")
    acc, base, null, chance = res
    ok = acc - null <= tol and acc - base <= tol and chance - acc <= tol
    check(out, name, round(acc - null, 4), ok, th, {"acc": round(acc, 4), "null": round(null, 4),
                                                     "majority": round(float(base), 4), "chance": round(chance, 4)})


def check_flip_stay(out, units):
    rows = defaultdict(list)
    for uid, u in units.items():
        o, e = shape(u["orig"]), shape(u["edit"])
        y, b = u["orig"]["m"]["label_flip"], u["orig"]["m"]["block"]
        for k, x in (("orig", o), ("edit", e), ("pair", o + e + [j - i for i, j in zip(o, e)])):
            rows[k].append((x, y, b, uid))
    for k in ("orig", "edit", "pair"):
        for mode in ("ood", "iid"):
            X, y, b, g = zip(*rows[k]) if rows[k] else ((),) * 4
            model_check(out, f"flip_stay.{k}.{mode}", fit_eval(X, y, b, g, mode) if rows[k] else None, 0.05)


def _bow_fit(Xa, ya, Xb):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = CountVectorizer(ngram_range=(1, 2), min_df=3, binary=True, max_features=200000)
    return LogisticRegression(max_iter=1000).fit(vec.fit_transform(Xa), ya).predict(vec.transform(Xb))


def _counts(it):
    """`shape` without the facts of the queried entity on the query path (also in the fact totals of the theory),
    then the facts of the queried entity off the path. The facts on the path are in `check_label_path`."""
    t, e, path = it["t"], it["q"]["entity"], query_path(it["t"], it["q"])
    t2 = {**t, "facts": [f for f in t["facts"] if not (f["entity"] == e and f["attr"] in path)]}
    return shape({**it, "t": t2})[:-4] + qsplit(it)[4:]


def check_label_cue(out, items, records):
    """Binary items at depth 1 or more. The counts probe uses `_counts` on the orig records. For the orig records the
    structure has no label signal of the semantics (`build` accepts a binary structure only if all 3 labels are
    possible). The edit records are in `check_label_path`: an edit can force a label signal in the structure, so they
    are compared with a label-blind reference. The counts and the text probes shuffle the null labels within each
    composition. The rule types in the text show the composition, and in the 7 compositions without a True-False
    flip, a flip from True or False must end in Unknown (edit records: Unknown 0.53 in train). A null with all labels
    shuffled cannot learn this share, so acc - null counted it as a cue. `steps.md` (Step 3) gives the label shares of
    these 7 compositions. The text probe uses depth 1 or more: at depth 0 the answer is a fact of the queried entity
    in the text. The depth probe uses all depths and only the depth, so that a label share that differs by depth is
    visible (a depth-0 flip cannot be Unknown, and a label controller can move the Unknown labels to the deeper
    units)."""
    rows = [(_counts(it), it["m"]["label"], it["m"]["block"], it["m"]["unit_id"], it["m"]["composition_id"],
             it["m"]["pair_role"]) for it in items.values() if it["q"]["type"] == "binary" and it["m"]["depth"] >= 1]
    dep = [([it["m"]["depth"]], it["m"]["label"], it["m"]["block"], it["m"]["unit_id"]) for it in items.values()
           if it["q"]["type"] == "binary"]
    recs = [r for r in records if r["task_type"] == "binary" and r["meta"]["depth"] >= 1]
    text = np.array([r["state"] + " || " + r["instruction"] for r in recs], dtype=object)
    for mode in ("ood", "iid"):
        sel = [r[:5] for r in rows if r[5] == "orig"]
        X, y, b, g, c = zip(*sel) if sel else ((),) * 5
        model_check(out, f"label_cue.counts_orig.{mode}", fit_eval(X, y, b, g, mode, strata=c) if sel else None, 0.03)
        X, y, b, g = zip(*dep) if dep else ((),) * 4
        model_check(out, f"label_cue.depth.{mode}", fit_eval(X, y, b, g, mode) if dep else None, 0.03)
        model_check(out, f"label_cue.bow.{mode}", fit_eval(text, [r["meta"]["label"] for r in recs],
                                                           [r["meta"]["block"] for r in recs],
                                                           [r["meta"]["unit_id"] for r in recs], mode, _bow_fit,
                                                           [r["meta"]["composition_id"] for r in recs])
                    if recs else None, 0.03)


def check_label_path(out, items, redraws=4):
    """The facts of the queried entity on the query path, binary items at depth 1 or more, for each role. Part of
    their signal is the semantics: a known label needs a rule whose conditions hold (in c00, a true fact on the path
    holds exactly when the label is known). Thus the probe is compared with a label-blind reference: the same theories
    with the facts on the path drawn again at random (`blind`, `redraws` times) and labeled with `solve`. Both sets
    keep the same number of each label in each block and composition (chance is 1/3 in both), and the reference keeps
    as many rows as the data. `path` uses the 4 counts on the path. `ref` uses `_counts` and these 4 counts, so it
    also covers the structure: an edit can force a label signal in the structure (a removed rule moves its facts off
    the path, and a removed rule on the query attribute removes one polarity), so the edit members are compared with
    the reference and not with a null. Pass if the accuracy on the data exceeds the accuracy on the reference by at
    most 0.03: the generator does not add a signal beyond the semantics. The nulls are not compared: with 4 discrete
    features, the null of a shuffled-label model varies too much (0.30 to 0.38 in a test). As in `model_check`, the
    accuracy must also be at least the chance agreement - 0.03: a lower accuracy shows a fact pattern that is
    inverted on the test mask (the search alone gave 0.29 in the val block, from c08). The detail gives the val
    composition with the lowest ood accuracy (about 450 rows each, so it is not a gate)."""
    rng = np.random.default_rng(SEED)
    th = "acc - acc of the blind reference <= 0.03, chance-acc <= 0.03"
    for role in ("orig", "edit"):
        real, ref = {"path": [], "ref": []}, {"path": [], "ref": []}
        for it in items.values():
            m = it["m"]
            if it["q"]["type"] != "binary" or m["pair_role"] != role or m["depth"] < 1 or it["s"] is None:
                continue
            key = (m["block"], m["unit_id"], m["composition_id"])
            c, p = _counts(it), qsplit(it)[:4]
            real["path"].append((p, m["label"]) + key)
            real["ref"].append((c + p, m["label"]) + key)
            for _ in range(redraws):
                t2, lab = blind(it, rng)
                if lab is not None:
                    p2 = qsplit({"t": t2, "q": it["q"]})[:4]
                    ref["path"].append((p2, lab) + key)
                    ref["ref"].append((c + p2, lab) + key)  # _counts has no facts on the path
        for kind in ("path", "ref"):
            rows, counts = _balanced(real[kind])
            refs, _ = _balanced(ref[kind], counts)  # the same number of rows as the data
            for mode in ("ood", "iid"):
                name, res, preds = f"label_cue.{kind}.{role}.{mode}", [], []

                def fit(Xa, ya, Xb):
                    pr = gbm().fit(Xa, ya).predict(Xb)
                    preds.append(pr)
                    return pr
                for rs in (rows, refs):
                    X, y, b, g, _ = zip(*rs) if rs else ((),) * 5
                    res.append(fit_eval(X, y, b, g, mode, fit) if rs else None)
                if None in res:
                    check(out, name, None, None, th, "no train or test data")
                    continue
                d = res[0][0] - res[1][0]
                detail = {"acc": round(res[0][0], 4), "chance": round(res[0][3], 4),
                          "blind_acc": round(res[1][0], 4), "rows": len(rows)}
                if mode == "ood":  # preds[0]: the first fit is the data (the 5 null fits follow)
                    te = [r for r in rows if r[2] == "val"]
                    ok = preds[0] == np.array([r[1] for r in te])
                    comp = np.array([r[4] for r in te])
                    worst = min(set(comp), key=lambda v: ok[comp == v].mean())
                    detail["worst_val"] = [str(worst), round(float(ok[comp == worst].mean()), 4)]
                check(out, name, round(d, 4), d <= 0.03 and res[0][3] - res[0][0] <= 0.03, th, detail)


def check_mc_cues(out, items):
    mc = [it for it in items.values() if it["q"]["type"] == "multi_choice" and it["m"]["depth"] >= 1]
    if not mc:
        return check(out, "mc_cues", None, None, "-", "no multi-choice item at depth >= 1")
    rule_acc, chance, uniq_ans, uniq_rand, by_depth = [], [], [], [], defaultdict(list)
    for it in mc:
        t, cands, tgt = it["t"], it["q"]["candidates"], it["rec"]["target"]
        n, hs = len(cands), [it["h"][a] for a in cands]
        mentioned = {c["attr"] for x in t["rules"] + t["exceptions"] for c in x["if"]} | \
                    {r["then"]["attr"] for r in t["rules"]}
        facts = {f["attr"] for f in t["facts"]}
        cand = [i for i, a in enumerate(cands) if a in mentioned and a not in facts] or list(range(n))
        rule_acc.append((tgt in cand) / len(cand))
        chance.append(1 / n)
        top = [i for i, x in enumerate(hs) if x == max(hs)]
        by_depth[it["m"]["depth"]].append(((tgt in top) / len(top), 1 / n))
        uniq_ans.append(len(top) == 1 and top[0] == tgt)
        uniq_rand.append((len(top) == 1) / n)
    v = float(np.mean(rule_acc) - np.mean(chance))
    check(out, "mc_cues.rules_not_state", round(v, 4), abs(v) <= 0.05, "|acc-chance|<=0.05",
          {"acc": round(float(np.mean(rule_acc)), 4), "chance": round(float(np.mean(chance)), 4)})
    gaps = {d: float(np.mean([a for a, _ in x]) - np.mean([c for _, c in x])) for d, x in sorted(by_depth.items())}
    worst = max(gaps.values(), key=abs)
    check(out, "mc_cues.max_height", round(worst, 4), abs(worst) <= 0.05, "|acc-chance|<=0.05 per depth",
          {d: round(g, 4) for d, g in gaps.items()})
    v = float(np.mean(uniq_ans) - np.mean(uniq_rand))
    check(out, "mc_cues.unique_max_height", round(v, 4), v <= 0.05, "P(ans)-P(rand)<=0.05",
          {"answer": round(float(np.mean(uniq_ans)), 4), "random": round(float(np.mean(uniq_rand)), 4)})
    tr = [it for it in mc if it["m"]["block"] == "train"]
    va = [it for it in mc if it["m"]["block"] == "val"]
    if not tr or not va:
        return check(out, "mc_cues.option_scorer", None, None, "|acc-chance|<=0.05", "no train or val data")
    X = [option_shape(it, a) for it in tr for a in it["q"]["candidates"]]
    y = [int(i == it["rec"]["target"]) for it in tr for i in range(len(it["q"]["candidates"]))]
    m = gbm().fit(np.array(X, dtype=float), y)
    hit = [int(np.argmax(m.predict_proba(np.array([option_shape(it, a) for a in it["q"]["candidates"]],
                                                  dtype=float))[:, 1]) == it["rec"]["target"]) for it in va]
    acc, ch = float(np.mean(hit)), float(np.mean([1 / len(it["q"]["candidates"]) for it in va]))
    check(out, "mc_cues.option_scorer", round(acc - ch, 4), abs(acc - ch) <= 0.05, "|acc-chance|<=0.05",
          {"acc": round(acc, 4), "chance": round(ch, 4)})


def check_polarity(out, items):
    for role in ("orig", "edit"):
        c = defaultdict(list)
        for it in items.values():
            m, q, t = it["m"], it["q"], it["t"]
            if q["type"] == "binary" and m["depth"] >= 1 and m["pair_role"] == role:
                negs = {r["then"]["neg"] for r in t["rules"] if r["then"]["attr"] == q["attr"]}
                c[m["label"]].append(len(negs) == 2)
        p = {k: round(float(np.mean(v)), 4) for k, v in sorted(c.items())}
        v = max(p.values()) - min(p.values()) if p else 0.0
        check(out, "polarity." + role, round(v, 4), v <= 0.05, "max-min<=0.05", p)


def check_inert_stays(out, units):
    n = inert = 0
    for u in units.values():
        o, e = u["orig"], u["edit"]
        if o["m"]["label_flip"] or (o["q"]["type"] == "binary" and o["m"]["depth"] == 0):
            continue
        if o["s"] is None or e["s"] is None:
            continue
        der = {r["then"]["attr"] for r in o["t"]["rules"] + e["t"]["rules"]}
        n += 1
        inert += all(o["s"].get(a, (None,))[0] == e["s"].get(a, (None,))[0] for a in der)
    v = inert / n if n else 0.0
    check(out, "inert_stays", round(v, 4), v <= 0.20, "<=0.20", {"stays": n, "inert": inert})


def check_pairs(out, units):
    cov, flips = Counter(), defaultdict(list)
    for u in units.values():
        m, me = u["orig"]["m"], u["edit"]["m"]
        bits = int(m["composition_id"][1:])
        flips[m["composition_id"]].append(m["label_flip"])
        if bits & 16 and not bits & 8 and m["label"] == "Unknown" and me["label"] != "Unknown":
            cov[m["edit_type"]] += 1
    c = {k: cov[k] for k in ("fact_remove", "priority_swap", "rule_remove")}
    check(out, "pair_coverage", min(c.values()), min(c.values()) > 0, ">0 each", c)
    share = {k: round(float(np.mean(v)), 4) for k, v in sorted(flips.items())}
    bad = {k: v for k, v in share.items() if not 0.50 <= v <= 0.70}
    lo, hi = (min(share.values()), max(share.values())) if share else (0, 0)
    check(out, "flip_share", f"{lo}..{hi}", not bad, "in [0.50,0.70]", bad or None)


def numbers(x):
    if isinstance(x, dict):
        for v in x.values():
            yield from numbers(v)
    elif isinstance(x, list):
        for v in x:
            yield from numbers(v)
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield x


def check_fill_rate(out, report):
    fr = report.get("fill_rate", report.get("targets", {}).get("fill_rate"))
    vals = list(numbers(fr)) if fr is not None else []
    if not vals:
        return check(out, "fill_rate", None, None, ">=0.95 each", "report.json has no fill_rate")
    check(out, "fill_rate", round(min(vals), 4), min(vals) >= 0.95, ">=0.95 each", {"cells": len(vals)})


def edit_ok(t, t2, path):
    """True if t2 is t with exactly one item edit (in place or deleted) at edit_path."""
    if t["entities"] != t2["entities"] or t["attributes"] != t2["attributes"]:
        return False
    changes = []
    for k in ("facts", "rules", "exceptions", "priorities"):
        a, b = t[k], t2[k]
        if len(a) == len(b):
            changes += [(k, i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
        elif len(a) == len(b) + 1:
            i = next((i for i in range(len(b)) if a[i] != b[i]), len(b))
            changes.append((k, i, a[i], None) if a[:i] + a[i + 1:] == b else (k, None, None, None))
        else:
            return False
    if len(changes) != 1 or changes[0][1] is None:
        return False
    k, i, x, y = changes[0]
    if path.split("/")[1:3] != [k, str(i)]:
        return False
    if y is None:
        return True
    if k == "priorities":
        return y == {"higher": x["lower"], "lower": x["higher"]}
    lx, ly = leaves(x), leaves(y)
    return lx.keys() == ly.keys() and sum(lx[p] != ly[p] for p in lx) == 1


def leaves(x, p=""):
    if isinstance(x, dict):
        return {k: v for key, val in x.items() for k, v in leaves(val, f"{p}/{key}").items()}
    if isinstance(x, list):
        return {k: v for j, val in enumerate(x) for k, v in leaves(val, f"{p}/{j}").items()}
    return {p: (type(x).__name__, x)}


def check_invariants(out, records, theories, items):
    bad = Counter()
    for h, t in theories.items():
        bad["theory_hash"] += canonical_hash(t) != h
    for it in items.values():
        m, r = it["m"], it["rec"]
        want = it["q"]["candidates"][r["target"]] if it["q"]["type"] == "multi_choice" else m["label"]
        bad["labels"] += (it["label"], it["depth"]) != (want, m["depth"]) or r["options"][r["target"]] != m["label"]
        bad["empty_rules"] += not it["t"]["rules"]
    pairs, groups, tq, text = defaultdict(dict), defaultdict(set), defaultdict(set), Counter()
    pos = defaultdict(Counter)
    for r in records:
        m = r["meta"]
        bad["theory_hash"] += m["theory_hash"] not in theories
        pairs[m["pair_id"]][m["pair_role"]] = r
        groups[m["paraphrase_group"]].add(json.dumps([m["theory_hash"], m["query"], m["label"], r["target"],
                                                      r["state"], r["options"], m["block"]], sort_keys=True))
        tq[(m["theory_hash"], json.dumps(m["query"], sort_keys=True))].add(m["paraphrase_group"])
        text[(r["state"], r["instruction"])] += 1
        p = PART.get(m["block"])
        tg = "tg%d" % (int(m["template_id"][1:]) // 3)
        bad["block_ids"] += p is None or m["composition_id"] not in p["comp"] or m["vocabulary_id"] not in p["voc"] \
            or m["template_group"] not in p["tg"] or tg != m["template_group"]
        bad["depth_range"] += p is None or m["depth"] not in p["depth"]
        if r["task_type"] == "multi_choice":
            n = len(r["options"])
            bad["mc_options"] += not 2 <= n <= 8 or len(m["query"]["candidates"]) != n or len(set(r["options"])) != n
            pos[n][r["target"]] += 1
    for pid, pr in pairs.items():
        o, e = pr.get("orig"), pr.get("edit")
        ok = o and e and o["meta"]["theory_hash"] in theories and e["meta"]["theory_hash"] in theories and \
            edit_ok(theories[o["meta"]["theory_hash"]], theories[e["meta"]["theory_hash"]], o["meta"]["edit_path"])
        bad["one_edit"] += not ok
    bad["paraphrase"] = sum(len(v) > 1 for v in groups.values())
    bad["dup_theory_query"] = sum(len(v) > 1 for v in tq.values())
    bad["dup_text"] = sum(v - 1 for v in text.values())
    for k in ("theory_hash", "labels", "one_edit", "block_ids", "depth_range", "dup_theory_query", "dup_text",
              "empty_rules", "mc_options", "paraphrase"):
        check(out, "invariants." + k, bad[k], bad[k] == 0, "==0")
    dev = max((abs(c[i] / sum(c.values()) - 1 / n) for n, c in pos.items() for i in range(n)), default=0.0)
    check(out, "invariants.mc_position", round(dev, 4), dev <= 0.03, "<=0.03",
          {n: sum(c.values()) for n, c in sorted(pos.items())})


def run(records, theories, report):
    items = build_items(records, theories)
    units = build_units(items)
    out = []
    check_flip_stay(out, units)
    check_label_cue(out, items, records)
    check_label_path(out, items)
    check_mc_cues(out, items)
    check_polarity(out, items)
    check_inert_stays(out, units)
    check_pairs(out, units)
    check_fill_rate(out, report)
    check_invariants(out, records, theories, items)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data_dir")
    args = ap.parse_args(argv)
    out = run(*load(args.data_dir))
    for c in out:
        print(f"{c['name']:30s} {str(c['value']):>14s}  {c['threshold']:30s} {c['status']}")
    failed = [c["name"] for c in out if c["status"] == "FAIL"]
    with open(Path(args.data_dir) / "audit.json", "w") as f:
        json.dump({"passed": not failed, "failed": failed, "checks": out}, f, indent=1, default=str)
    print("FAIL: " + ", ".join(failed) if failed else "PASS: all checks")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
