"""Step 4: build, check, freeze, and load the splits. See docs/splits.md.

Build the files and the manifest: python -m crux.splits build [--refreeze]
Check the files against the manifest: python -m crux.splits verify
Count again the Unknown-license tasks of Super-NaturalInstructions: python -m crux.splits sni-recount

The splits (data/splits/, git-ignored): train, dev_id, validation, test, and the evaluation-only benchmarks in
benchmarks/. configs/splits_manifest.json (in git) holds the SHA-256, the counts, the leakage report, and the baselines.
Settings: configs/step4_splits.yaml. Load a frozen split with `load_split`.
"""
import argparse
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np
import yaml

from crux import metrics
from crux.schema import PROCESSED_DIR, ROOT, read_jsonl, summarize, write_jsonl

CONFIG = ROOT / "configs" / "step4_splits.yaml"
MANIFEST = ROOT / "configs" / "splits_manifest.json"
PROTOCOL = ROOT / "configs" / "protocol.yaml"
SPLITS = ("train", "dev_id", "validation", "test")
BENCHMARKS = ("boardgameqa", "dynabench", "supernat")
# pair_id and paraphrase_group are derived from unit_id (spec 4.5), so unit_id covers pair_id. paraphrase_group stays
# in the list because the protocol names paraphrases that do not occur in train.
ID_KEYS = ("template_id", "template_group", "composition_id", "vocabulary_id", "paraphrase_group", "unit_id", "theory_hash")
ID_PAIRS = (("train", "validation", ID_KEYS), ("train", "test", ID_KEYS), ("validation", "test", ID_KEYS),
            ("dev_id", "validation", ID_KEYS), ("dev_id", "test", ID_KEYS),
            ("train", "dev_id", ("unit_id",)))  # dev_id is in-distribution: it shares everything with train but the units
# (record set, reference). A record is a near-duplicate if too many of its n-grams occur in the reference.
NGRAM_PAIRS = (("validation", "train"), ("test", "train"), ("boardgameqa", "train"), ("dynabench", "train"),
               ("supernat", "train"), ("test", "validation"))
RULE_EDITS = {"exception_remove", "priority_swap", "rule_remove", "condition_negate", "threshold_shift"}
TOKEN_RE = re.compile(r"[a-z0-9]+")
MULT = np.uint64(0x9E3779B97F4A7C15)


def unit_key(rec):
    """Records with the same key stay together: a synthetic unit, or one record."""
    return rec["meta"].get("unit_id", rec["id"])


def in_dev(unit_id, seed, share):
    """True for about `share` of the units: a seeded hash of the unit id, so the choice does not depend on the order."""
    return int(hashlib.sha256(f"{seed}/{unit_id}".encode()).hexdigest()[:16], 16) / 16**16 < share


def split_train_dev(records, seed, share):
    """Split the train-block records into (train, dev_id) by unit. All records of a unit go to the same side."""
    dev = {unit_key(r) for r in records if in_dev(unit_key(r), seed, share)}
    return [r for r in records if unit_key(r) not in dev], [r for r in records if unit_key(r) in dev]


def cap_per_task(records, cap):
    """Keep the first `cap` records of each task in file order. The official Super-NaturalInstructions test
    split takes the first 100 instances of a task (Tk-Instruct `ni_dataset.py`), so the first 100 match it."""
    seen, out = Counter(), []
    for r in records:
        seen[r["meta"]["task_id"]] += 1
        if seen[r["meta"]["task_id"]] <= cap:
            out.append(r)
    return out


# ---------------------------------------------------------------- leakage

def check_ids(splits):
    """Raise ValueError if two splits share an id of ID_PAIRS."""
    bad = {}
    for a, b, keys in ID_PAIRS:
        for key in keys:
            shared = {r["meta"][key] for r in splits[a] if key in r["meta"]} & \
                     {r["meta"][key] for r in splits[b] if key in r["meta"]}
            if shared:
                bad[f"{a}~{b}:{key}"] = f"{len(shared)} shared, for example {sorted(shared)[:3]}"
    if bad:
        raise ValueError(f"shared ids between splits: {bad}")


@lru_cache(maxsize=None)
def _token_hash(token):
    return int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")


def grams(rec, n):
    """64-bit hashes of the word n-grams of state + instruction (lowercase, tokens [a-z0-9]+)."""
    tokens = TOKEN_RE.findall((rec["state"] + " " + rec["instruction"]).lower())
    m = len(tokens) - n + 1
    if m <= 0:
        return np.empty(0, np.uint64)
    t = np.array([_token_hash(x) for x in tokens], dtype=np.uint64)
    h = np.zeros(m, np.uint64)
    for j in range(n):  # polynomial hash of n token hashes; the uint64 product wraps
        h = h * MULT + t[j:j + m]
    return h


def shares(ref, arrays):
    """For each array: the share of its n-grams that occur in the sorted unique array `ref`."""
    out = np.zeros(len(arrays))
    for i, a in enumerate(arrays):
        if len(a) and len(ref):
            out[i] = np.mean(ref[np.minimum(np.searchsorted(ref, a), len(ref) - 1)] == a)
    return out


def leakage(splits, n, max_overlap, rounds=5):
    """Check the splits, remove the leaking records, and check again. Return (clean splits, report).

    Ids: shared ids raise an error (check_ids). Exact duplicates: a (state, instruction) pair in two splits is
    removed from the later split of SPLITS + BENCHMARKS. Near-duplicates: a record of an evaluation split is
    removed if the share of its n-grams in the reference of NGRAM_PAIRS is more than `max_overlap`.
    A removed synthetic record takes its whole unit with it. The report holds the overlap distribution of
    the first round, and the removed ids.
    """
    splits = {k: list(v) for k, v in splits.items()}
    arrs = {k: [grams(r, n) for r in v] for k, v in splits.items() if any(k == p[0] or k == p[1] for p in NGRAM_PAIRS)}
    train_ref = np.unique(np.concatenate(arrs["train"] + [np.empty(0, np.uint64)]))
    del arrs["train"]
    report = {"ngram": {"n": n, "max_overlap": max_overlap, "pairs": {}}, "removed": {}}
    for rnd in range(rounds):
        check_ids(splits)
        flags = defaultdict(lambda: defaultdict(set))  # split -> reason -> unit keys
        seen = {}
        for name in SPLITS + BENCHMARKS:
            for r in splits[name]:
                key = hashlib.sha1((r["state"] + "\x00" + r["instruction"]).encode()).digest()
                if seen.setdefault(key, name) != name:
                    flags[name]["exact"].add(unit_key(r))
        val_ref = np.unique(np.concatenate(arrs["validation"] + [np.empty(0, np.uint64)]))
        for name, ref_name in NGRAM_PAIRS:
            s = shares(train_ref if ref_name == "train" else val_ref, arrs[name])
            over = s > max_overlap
            if rnd == 0:
                report["ngram"]["pairs"][f"{name}~{ref_name}"] = {
                    "n": len(s), "max": round(float(s.max(initial=0)), 4),
                    "p99": round(float(np.percentile(s, 99)) if len(s) else 0, 4), "over": int(over.sum())}
            flags[name]["ngram"] |= {unit_key(r) for r, o in zip(splits[name], over) if o}
        if not any(keys for reasons in flags.values() for keys in reasons.values()):
            report["rounds"] = rnd + 1
            return splits, report
        for name, reasons in flags.items():
            drop = set().union(*reasons.values())
            if not drop:
                continue
            gone = sorted(r["id"] for r in splits[name] if unit_key(r) in drop)
            entry = report["removed"].setdefault(name, {"exact": 0, "ngram": 0, "ids": []})
            for reason, keys in reasons.items():
                entry[reason] += len(keys)
            entry["ids"] = sorted(entry["ids"] + gone)
            keep = [i for i, r in enumerate(splits[name]) if unit_key(r) not in drop]
            splits[name] = [splits[name][i] for i in keep]
            if name in arrs:
                arrs[name] = [arrs[name][i] for i in keep]
    raise RuntimeError(f"the leakage check does not converge in {rounds} rounds")


# ---------------------------------------------------------------- baselines

STATS = ("train", "dev_id", "validation")  # the splits with label and model statistics. Step 8 computes them for test.


def baselines(splits, theories):
    """Reference values, not model results: the majority-class baseline, the edit pairs, and the fact-count baseline.
    A model must score above them (accuracy, higher is better). `theories` maps theory_hash to the theory.
    The protocol evaluates the test split once, in Step 8. Thus no label or model statistic of test is here."""
    train = splits["train"]
    out = {"majority": {}, "composition": {}, "pairs": {}}
    for name in STATS + BENCHMARKS:
        parts = defaultdict(list)
        for r in splits[name]:
            tag = {"proofwriter": r["meta"].get("subset"), "synthetic": r["task_type"]}.get(r["source"])
            parts[r["source"] + ("/" + tag if tag else "")].append(r)
        ref = train if name in STATS else None  # a benchmark has its own labels: its majority comes from its own records
        out["majority"][name] = {p: {"n": len(rs), "majority": round(metrics.majority_baseline(rs, ref=ref)["majority_baseline"], 4)}
                                 for p, rs in sorted(parts.items())}
    for name in STATS:
        comp, pairs, units = {}, {t: Counter() for t in ("binary", "multi_choice")}, defaultdict(set)
        for role in ("orig", "edit"):
            binary = lambda records: [r for r in records if r["source"] == "synthetic" and r["task_type"] == "binary"
                                      and r["meta"]["pair_role"] == role]
            rs, groups = binary(splits[name]), defaultdict(list)
            acc = metrics.majority_baseline(rs, by="composition_id", ref=binary(train))  # the train majority of the group
            for r in rs:
                groups[r["meta"]["composition_id"]].append(r)
            for c, g in groups.items():
                comp[f"{c}/{role}"] = {"n": len(g), "majority": round(acc[f"majority_baseline/composition_id={c}"], 4),
                                       "unknown": round(sum(r["target"] == 2 for r in g) / len(g), 4)}
        for r in splits[name]:
            m = r["meta"]
            if r["source"] == "synthetic" and m["pair_role"] == "orig":
                c = pairs[r["task_type"]]
                c["pairs"] += 1
                c["flips" if m["label_flip"] else ("relevant_stays" if m["edit_relevant"] else "irrelevant_stays")] += 1
                if m["edit_type"] in RULE_EDITS:
                    c["rule_edit_pairs"] += 1
                    units[r["task_type"]].add(m["unit_id"])
        for t, c in pairs.items():
            c["rule_edit_units"] = len(units[t])
        out["composition"][name] = dict(sorted(comp.items()))
        out["pairs"][name] = {t: dict(c) for t, c in pairs.items()}
    out["fact_count"] = fact_count(splits, theories)
    return out


def fact_count(splits, theories):
    """The Step 3 fact-count baseline on the frozen splits. Reuse the probe of crux/gen/audit.py: a gradient-boosting
    classifier on the true, false, and missing facts of the queried entity on the query path. It trains on the
    orig binary records of depth 1 or more in train, balanced by label in each composition. It scores the balanced
    orig records of dev_id and validation (accuracy; chance is about 1/3). It does not read test."""
    from crux.gen import audit

    recs = [r for k in STATS for r in splits[k]
            if r["source"] == "synthetic" and r["task_type"] == "binary" and r["meta"]["pair_role"] == "orig"
            and r["meta"]["depth"] >= 1]
    where = {r["id"]: k for k in STATS for r in splits[k]}
    rows = [(audit.qsplit(it)[:4], it["m"]["label"], where[it["rec"]["id"]], it["m"]["unit_id"], it["m"]["composition_id"])
            for it in audit.build_items(recs, theories).values() if it["q"]["type"] == "binary" and it["s"] is not None]
    rows, _ = audit._balanced(rows)
    fit = [r for r in rows if r[2] == "train"]
    model = audit.gbm().fit(np.array([r[0] for r in fit], dtype=float), [r[1] for r in fit])
    out = {}
    for k in STATS[1:]:  # the accuracy on train is in-sample
        te = [r for r in rows if r[2] == k]
        if te:
            pred = model.predict(np.array([r[0] for r in te], dtype=float))
            out[k] = {"n": len(te), "accuracy": round(float(np.mean(pred == np.array([r[1] for r in te]))), 4)}
    return out


# ---------------------------------------------------------------- build, freeze, load

def sha256_of(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sni_recount(repo_dir):
    """Count the held-out classification tasks of Super-NaturalInstructions with instance license "Unknown"."""
    from crux.data import supernat

    tasks = [t for _, t in supernat.load_tasks(supernat.download(repo_dir))]
    cls = [t for t in tasks if supernat.is_classification_category(t["Categories"]) and supernat.label_set(t) is not None]
    return {"heldout_tasks": len(tasks), "classification_tasks": len(cls),
            "unknown_license": sum("Unknown" in t["Instance License"] for t in cls),
            "unknown_license_all_heldout": sum("Unknown" in t["Instance License"] for t in tasks),
            "selected": sum(supernat.select_task(t)[0] is not None for t in tasks)}


def collect(cfg):
    """Return the splits before the leakage check, and the theories of the synthetic data."""
    from crux.data import proofwriter
    from crux.gen import audit

    synthetic, theories = audit.load(ROOT / cfg["synthetic_dir"])[:2]
    blocks = defaultdict(list)
    for r in synthetic:
        blocks[r["meta"]["block"]].append(r)
    train, dev = split_train_dev(blocks["train"], cfg["seed"], cfg["dev_share"])
    pw = [r for s in proofwriter.TEST_SUBSETS
          for r in proofwriter.to_test_records(proofwriter.download_test(ROOT / cfg["proofwriter_cache"], s), s)]
    bqa = [r for d in sorted((PROCESSED_DIR / "boardgameqa").iterdir()) for s in cfg["boardgameqa_splits"]
           for r in read_jsonl(d / f"{s}.jsonl")]
    sni = cap_per_task(read_jsonl(PROCESSED_DIR / "supernat" / "test.jsonl"), cfg["supernat_cap"])
    # docs/contamination.md: remove the test records that occur in the Ettin pretraining data before the freeze.
    lines = (ROOT / cfg["contamination_removed"]).read_text().splitlines()
    drop = {s.strip() for s in lines if s.strip() and not s.startswith("#")}
    test = [r for r in blocks["test"] + pw if r["id"] not in drop]
    if len(test) != len(blocks["test"]) + len(pw) - len(drop):
        raise ValueError(f"{cfg['contamination_removed']}: some ids are not in the test split")
    splits = {"train": train, "dev_id": dev,
              "validation": blocks["val"], "test": test, "boardgameqa": bqa,
              "dynabench": read_jsonl(PROCESSED_DIR / "dynabench" / "test.jsonl"), "supernat": sni}
    return splits, theories


def write_splits(splits, out, final=None):
    """Write each split sorted by id (so the bytes do not depend on the input order). Return the manifest entries.
    `final` is the folder that build moves the files to. The entry paths name it (default `out`). The test entry
    has no label counts: the protocol evaluates the test split once, in Step 8."""
    final, entries = final or out, {}
    for name, records in splits.items():
        path = write_jsonl(sorted(records, key=lambda r: r["id"]), out / ("benchmarks" if name in BENCHMARKS else "") / f"{name}.jsonl")
        entries[name] = {"path": str((final / path.relative_to(out)).relative_to(ROOT)), "sha256": sha256_of(path), **summarize(records)}
        if name == "test":
            del entries[name]["label"]
        subsets = Counter(r["meta"]["subset"] for r in records if "subset" in r["meta"])
        if subsets:
            entries[name]["subset"] = dict(sorted(subsets.items()))
    return entries


def build(refreeze=False):
    """Write the files to `<out_dir>.new`, compare their SHA-256 with the manifest, and replace the frozen files
    only if nothing changed or `refreeze` is set. A refused build leaves the frozen files and the manifest as they are."""
    cfg = yaml.safe_load(CONFIG.read_text())
    splits, theories = collect(cfg)
    splits, leaks = leakage(splits, cfg["ngram_n"], cfg["max_overlap"])
    out = ROOT / cfg["out_dir"]
    new = out.with_name(out.name + ".new")
    shutil.rmtree(new, ignore_errors=True)
    entries = write_splits(splits, new, out)
    prev = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    changed = [k for k, v in prev.get("splits", {}).items() if v["sha256"] != entries.get(k, {}).get("sha256")]
    if changed and not refreeze:
        shutil.rmtree(new)
        raise SystemExit(f"the splits {changed} differ from the frozen manifest. The frozen files stay. Use --refreeze only before Step 5.")
    for path in sorted(new.rglob("*.jsonl")):  # a file with the same bytes stays where it is
        dest = out / path.relative_to(new)
        if not dest.exists() or sha256_of(dest) != sha256_of(path):
            dest.parent.mkdir(parents=True, exist_ok=True)
            path.replace(dest)
    shutil.rmtree(new)
    manifest = {"settings": {k: cfg[k] for k in ("seed", "dev_share", "ngram_n", "max_overlap", "supernat_cap", "boardgameqa_splits")},
                "splits": entries, "leakage": leaks, "baselines": baselines(splits, theories),
                "contamination_removed": {"file": cfg["contamination_removed"],
                                          "sha256": sha256_of(ROOT / cfg["contamination_removed"])},
                "supernat": prev.get("supernat")}  # recorded once by `sni-recount` (docs/splits.md)
    MANIFEST.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    return manifest


def load_split(name, allow_test=False):
    """Return the records of a frozen split, after a SHA-256 check against the manifest and, for train, validation,
    and test, against the hash that `configs/protocol.yaml` pre-registers.

    The protocol forbids reading the test split outside Step 8. Only Step 8 sets allow_test=True.
    """
    if name == "test" and not allow_test:
        raise PermissionError("the test split is for Step 8 only. Set allow_test=True there.")
    entry = json.loads(MANIFEST.read_text())["splits"][name]
    path = ROOT / entry["path"]
    digest = sha256_of(path)
    anchors = {"the manifest": entry["sha256"],
               "protocol.yaml": yaml.safe_load(PROTOCOL.read_text())["splits"]["split_hash"].get(name, entry["sha256"])}
    bad = [k for k, v in anchors.items() if v != digest]
    if bad:
        raise ValueError(f"SHA-256 mismatch for {path}: the split differs from {' and '.join(bad)}, so it changed after the freeze")
    return read_jsonl(path)


def verify():
    """Check the SHA-256 of every file against the manifest. It hashes the bytes and does not parse the records."""
    for name, entry in json.loads(MANIFEST.read_text())["splits"].items():
        path = ROOT / entry["path"]
        ok = path.exists() and sha256_of(path) == entry["sha256"]
        print(f"{'OK  ' if ok else 'FAIL'} {name:12s} {entry['n']:>8d} {entry['sha256'][:16]}")
        if not ok:
            raise SystemExit(f"{path}: missing or different from the manifest")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["build", "verify", "sni-recount"])
    ap.add_argument("--refreeze", action="store_true", help="accept changed hashes (only before Step 5)")
    args = ap.parse_args()
    if args.command == "build":
        m = build(args.refreeze)
        print(json.dumps({"splits": {k: [v["n"], v["sha256"]] for k, v in m["splits"].items()},
                          "ngram": m["leakage"]["ngram"]["pairs"], "removed": {k: len(v["ids"]) for k, v in m["leakage"]["removed"].items()}}, indent=1))
    elif args.command == "verify":
        verify()
    else:
        print(json.dumps(sni_recount(ROOT / yaml.safe_load(CONFIG.read_text())["supernat_repo"]), indent=1))


if __name__ == "__main__":
    main()
