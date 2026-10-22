"""Step 5: select the trial of each architecture and compare it with A.

Run: python -m crux.aggregate runs/step5
Reads <root>/<arch>/t<trial>/s<seed_index>/metrics.json and val_probs.npz. Only a trial with all seeds counts.
Per architecture it selects the trial with the best mean validation `--metric` over the seeds. It prints the
mean +- std (ddof 1) over the seeds, and the paired bootstrap CI of the difference to A (configs/protocol.yaml
`statistics`: resamples, seed, ci_level). The bootstrap runs for the two primary metrics of the protocol:
unseen_rule_accuracy (the unit is the record with unseen ids) and counterfactual_paired_accuracy (the unit is the
pair: orig and edit both correct), and for accuracy. The statistic is the mean difference of the per-unit hits over
the seed pairs (the same seed index of A and of the architecture).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from crux.metrics import _leaked, seen_ids
from crux.splits import load_split

ROOT = Path(__file__).resolve().parents[1]
SHOWN = ("unseen_rule_accuracy", "paired_accuracy", "accuracy", "paraphrase_consistency", "ece")
LOWER_IS_BETTER = {"ece"}


def load_runs(root):
    """Return {arch: {trial: {seed_index: metrics}}}. A run of Steps 5 and 6 has no order invariance or memory in
    metrics.json: they come from order_invariance.json (`python -m crux.train --eval-only`) if the file exists."""
    runs = defaultdict(lambda: defaultdict(dict))
    for f in sorted(Path(root).glob("*/t*/s*/metrics.json")):
        seed_dir, trial_dir, arch_dir = f.parent, f.parent.parent, f.parent.parent.parent
        m = json.loads(f.read_text())
        if (f.parent / "order_invariance.json").exists() and "order_invariance" not in m:
            extra = json.loads((f.parent / "order_invariance.json").read_text())
            m.update({k: extra[k] for k in ("order_invariance", "memory")})
        runs[arch_dir.name][int(trial_dir.name[1:])][int(seed_dir.name[1:])] = m
    return runs


def mean_std(values):
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")


def dig(m, *keys):
    """m[k1][k2]... or None if a key is missing (an old run) or the value is null."""
    for k in keys:
        m = m.get(k) if isinstance(m, dict) else None
    return m


def mean_or_none(ms, *keys):
    """Mean over the runs of the value at `keys`, or None if a run has no value."""
    values = [dig(m, *keys) for m in ms]
    return None if any(v is None for v in values) else float(np.mean(values))


# Step 7 columns: (title, direction, keys in metrics.json, divisor). The divisor turns bytes into MB or KB.
COST_COLUMNS = (
    ("order invariance: argmax agreement", "higher", ("order_invariance", "argmax_agreement"), 1),
    ("order invariance: mean abs prob diff", "lower", ("order_invariance", "mean_abs_prob_diff"), 1),
    ("unseen_rule_accuracy, shuffled options", "higher", ("order_invariance", "shuffled", "unseen_rule_accuracy"), 1),
    ("model MB fp16", "lower", ("memory", "model_bytes", "fp16"), 1e6),
    ("peak CUDA MB above model, bs1", "lower", ("memory", "cuda_peak_extra_bytes", "batch_size_1"), 1e6),
    ("peak CUDA MB above model, batches", "lower", ("memory", "cuda_peak_extra_bytes", "eval_batches"), 1e6),
    ("cache KB per state", "lower", ("memory", "cache_bytes_fp16", "state_mean"), 1e3),
    ("cache KB per instruction", "lower", ("memory", "cache_bytes_fp16", "instruction_mean"), 1e3),
    ("bs1 p50 ms, state cached", "lower", ("latency", "cache_hit", "state", "batch_size_1", "p50_ms_per_record"), 1),
    ("bs1 p50 ms, state and instruction cached", "lower",
     ("latency", "cache_hit", "state_and_instruction", "batch_size_1", "p50_ms_per_record"), 1),
)


def bootstrap_ci(hits_x, hits_a, resamples, seed, level, chunk=250):
    """hits_*: [seeds, n] arrays of 0/1. Return (mean difference, low, high). Paired by example and by seed."""
    d = (hits_x.astype(float) - hits_a.astype(float)).mean(0)
    rng, n, means = np.random.default_rng(seed), d.shape[0], []
    for _ in range(0, resamples, chunk):
        means.append(d[rng.integers(0, n, (min(chunk, resamples), n))].mean(1))
    lo, hi = np.percentile(np.concatenate(means), [(1 - level) * 50, 100 - (1 - level) * 50])
    return float(d.mean()), float(lo), float(hi)


def pair_index(records):
    """Indices (orig, edit) of the pairs of `records` (`meta.pair_id`), as two lists of the same length."""
    where = {}
    for i, r in enumerate(records):
        pid = r["meta"].get("pair_id")
        if pid is not None:
            where.setdefault(pid, {})[r["meta"]["pair_role"]] = i
    return [d["orig"] for d in where.values()], [d["edit"] for d in where.values()]


def pair_hits(records, hits):
    """hits [seeds, n] -> [seeds, pairs]: 1 if the orig and the edit record of a pair are both hits."""
    o, e = pair_index(records)
    return hits[:, o] * hits[:, e]


def select(runs, metric, n_seeds):
    """Per architecture: the trial with all seeds and the best mean validation metric."""
    chosen = {}
    for arch, trials in runs.items():
        full = {t: s for t, s in trials.items() if len(s) == n_seeds}
        if full:
            chosen[arch] = max(full, key=lambda t: np.mean([m["validation"][metric] for m in full[t].values()]))
    return chosen


def hits(run_dir):
    return np.load(run_dir / "val_probs.npz")["hits"]


def aggregate(root, metric="unseen_rule_accuracy", ref="A"):
    """`ref`: the reference of the paired bootstrap (the protocol uses A; Step 6 also compares with B and B+CF)."""
    root = Path(root)
    protocol = yaml.safe_load((ROOT / "configs" / "protocol.yaml").read_text())
    stats = protocol["statistics"]
    runs, n_seeds = load_runs(root), len(protocol["seeds"])
    chosen = select(runs, metric, n_seeds)
    if "A" not in chosen or ref not in chosen:
        raise SystemExit(f"A or {ref} has no trial with all seeds.")
    out = {"metric": metric, "selected": chosen, "architectures": {}, f"difference_to_{ref}": {}}
    print(f"Selection: the trial with the best mean validation {metric} over {n_seeds} seeds (higher is better).")
    for arch, trials in sorted(runs.items()):
        incomplete = sorted(t for t, s in trials.items() if len(s) < n_seeds)
        print(f"{arch}: trials with all seeds {sorted(set(trials) - set(incomplete))}, incomplete {incomplete}")
    rows = {}
    for arch, t in chosen.items():
        ms = [runs[arch][t][s] for s in sorted(runs[arch][t])]
        row = {k: mean_std([m["validation"][k] for m in ms]) for k in SHOWN if k in ms[0]["validation"]}
        row["dev_id_accuracy"] = mean_std([m["dev_id"]["accuracy"] for m in ms])
        row["majority_baseline"] = ms[0]["validation"]["majority_baseline"]
        row["dev_id_majority_baseline"] = ms[0]["dev_id"]["majority_baseline"]
        row["params"] = {k: ms[0][k] for k in ("params_total", "params_trainable", "params_extra", "static_target")}
        row["latency"] = {f"{k}/{p}": mean_std([m["latency"][k][p] for m in ms])[0] for k in ("batch_size_1", "eval_batches")
                          for p in ("p50_ms_per_record", "p95_ms_per_record")}
        row["cost"] = {title: mean_or_none(ms, *keys) for title, _, keys, _ in COST_COLUMNS}
        rows[arch] = row
    out["architectures"] = rows
    print("\nValidation, mean +- std over seeds (the direction is in each column)")
    print("| arch | trial | " + " | ".join(f"{k} ({'lower' if k in LOWER_IS_BETTER else 'higher'} is better)" for k in SHOWN)
          + " | dev_id accuracy (higher is better) | majority baseline (reference) | dev_id majority baseline (reference) |")
    print("|---|---|" + "---|" * (len(SHOWN) + 3))
    for arch, r in rows.items():
        cells = [f"{r[k][0]:.4f} +- {r[k][1]:.4f}" for k in SHOWN if k in r]
        print(f"| {arch} | {chosen[arch]} | " + " | ".join(cells) +
              f" | {r['dev_id_accuracy'][0]:.4f} +- {r['dev_id_accuracy'][1]:.4f} | {r['majority_baseline']:.4f} | {r['dev_id_majority_baseline']:.4f} |")
    print("\nParameter counts and latency per record, mean over seeds")
    print("| arch | params_total | params_trainable | extra vs A | static target | bs1 p50 ms (lower is better) | bs1 p95 ms (lower is better) | batches p50 ms (lower is better) | batches p95 ms (lower is better) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for arch, r in rows.items():
        p, lat = r["params"], r["latency"]
        print(f"| {arch} | {p['params_total']} | {p['params_trainable']} | {p['params_extra']} | {p['static_target']} | "
              + " | ".join(f"{v:.2f}" for v in lat.values()) + " |")
    print("\nOrder invariance (validation, shuffled options), memory, and cache-hit latency per record, mean over seeds "
          "(n/a: the run has no value)")
    print("| arch | " + " | ".join(f"{t} ({d} is better)" for t, d, _, _ in COST_COLUMNS) + " |")
    print("|---|" + "---|" * len(COST_COLUMNS))
    for arch, r in rows.items():
        cells = [("n/a" if r["cost"][t] is None else f"{r['cost'][t] / k:.4g}") for t, _, _, k in COST_COLUMNS]
        print(f"| {arch} | " + " | ".join(cells) + " |")
    if "B" in rows and "A-param" in rows:
        gap = rows["A-param"]["params"]["params_trainable"] - rows["B"]["params"]["params_trainable"]
        print(f"\nA-param minus B trainable parameters: {gap}")
        out["a_param_gap"] = gap
    seeds = sorted(runs[ref][chosen[ref]])
    val = load_split("validation")
    seen = seen_ids(load_split("train"))
    unseen = [i for i, r in enumerate(val) if not _leaked(r, seen)]
    units = {"unseen_rule_accuracy": lambda h: h[:, unseen], "counterfactual_paired_accuracy": lambda h: pair_hits(val, h),
             "accuracy": lambda h: h}
    hits_a = np.stack([hits(root / ref / f"t{chosen[ref]}" / f"s{s}") for s in seeds])
    print(f"\nDifference to {ref}, {stats['bootstrap']['ci_level']:.0%} paired bootstrap CI "
          f"({stats['bootstrap']['resamples']} resamples, mean over {len(seeds)} seed pairs; higher is better)")
    print("| arch | metric | difference | CI low | CI high | CI excludes 0 |")
    print("|---|---|---|---|---|---|")
    for arch, t in chosen.items():
        if arch == ref:
            continue
        if sorted(runs[arch][t]) != seeds:
            raise SystemExit(f"{arch} and {ref} need the same seed indices.")
        hx = np.stack([hits(root / arch / f"t{t}" / f"s{s}") for s in seeds])
        for name, unit in units.items():
            d, lo, hi = bootstrap_ci(unit(hx), unit(hits_a), stats["bootstrap"]["resamples"], stats["bootstrap"]["seed"],
                                     stats["bootstrap"]["ci_level"])
            out[f"difference_to_{ref}"].setdefault(arch, {})[name] = {"difference": d, "ci_low": lo, "ci_high": hi,
                                                                 "excludes_0": bool(lo > 0 or hi < 0)}
            print(f"| {arch} | {name} | {d:+.4f} | {lo:+.4f} | {hi:+.4f} | {'yes' if lo > 0 or hi < 0 else 'no'} |")
    (root / ("aggregate.json" if ref == "A" else f"aggregate_ref_{ref}.json")).write_text(json.dumps(out, indent=1, default=float))
    return out


# ---------------------------------------------------------------- Step 8: the benchmark report

LOWER = ("ece",)
TABLE_KEYS = ("accuracy", "macro_f1", "ece", "unseen_rule_accuracy", "paired_accuracy", "paraphrase_consistency")


def _stat(entry, group, many, pooled):
    """The metrics of one level of a bench.json set entry, without the group prefix."""
    if pooled:
        return dict(entry["pooled"])
    return {k[len(group) + 1:]: v for k, v in entry["metrics"].items() if k.startswith(group + "/")} if many else dict(entry["metrics"])


def bench_report(config="configs/step8_eval.yaml", out=None):
    """Step 8: read bench.json and probs_<set>.npz of every model with 3 complete seeds, and write the report
    (<report>.json, <report>.txt). For each set and group: mean +- std over the seeds, the majority baseline, and the paired
    bootstrap CI of the difference to the reference (A) for the accuracy, the unseen-rule accuracy, and the counterfactual
    paired accuracy (the unit is the record, the record with unseen ids, and the pair). `out` replaces out_root (tests)."""
    cfg = yaml.safe_load((ROOT / config).read_text())
    protocol = yaml.safe_load((ROOT / "configs" / "protocol.yaml").read_text())
    b, n_seeds, ref = protocol["statistics"]["bootstrap"], len(protocol["seeds"]), cfg["reference"]
    out_dir = Path(out) if out else ROOT / cfg["out_root"]
    bench, npz = {}, {}
    for name in cfg["models"]:
        dirs = [out_dir / name / f"s{k}" for k in range(n_seeds)]
        if all((d / "bench.json").exists() for d in dirs):
            bench[name] = [json.loads((d / "bench.json").read_text()) for d in dirs]
            npz[name] = dirs
    sets = [s for s in cfg["sets"] if bench and all(s in x["sets"] for v in bench.values() for x in v)]
    if ref not in bench or not sets:
        raise SystemExit(f"{ref} or the sets are missing in {out_dir}.")
    base = json.loads((out_dir / "baselines.json").read_text()) if (out_dir / "baselines.json").exists() else {}
    res = {"reference": ref, "sets": sets, "models": {}, "levels": {}, "baselines": base,
           "bootstrap": b, "eval_limits": bench[ref][0]["eval_limits"]}
    for name, runs in bench.items():  # parameters, and the L4 latency of the training run
        train = [json.loads(f.read_text()) for f in sorted((ROOT / cfg["models"][name]["root"] / name / "t0").glob("s*/metrics.json"))]
        res["models"][name] = {"family": cfg["models"][name]["family"], "seeds": [x["seed"] for x in runs],
                               "params_total": runs[0]["params_total"], "params_trainable": runs[0]["params_trainable"],
                               "latency_bs1_p50_ms": mean_or_none(train, "latency", "batch_size_1", "p50_ms_per_record"),
                               "latency_bs1_p95_ms": mean_or_none(train, "latency", "batch_size_1", "p95_ms_per_record")}
    for s in sets:
        arr = {n: [np.load(d / f"probs_{s}.npz") for d in dirs] for n, dirs in npz.items()}
        r0 = arr[ref][0]
        if any(not np.array_equal(a["ids"], r0["ids"]) for v in arr.values() for a in v):
            raise SystemExit(f"{s}: the models have different record ids.")
        hits = {n: np.stack([a["hits"] for a in v]) for n, v in arr.items()}
        groups = sorted(set(r0["group"]))
        many = len(groups) > 1
        levels = [(g, np.flatnonzero(r0["group"] == g), False) for g in groups]
        if many and s != "test":  # the pooled value of benchmarks (a pool of test sources is not a result)
            levels.append(("pooled", np.arange(len(r0["ids"])), True))
        po, pe = r0["pair_orig"], r0["pair_edit"]
        for g, idx, is_pool in levels:
            sel = np.isin(po, idx)
            units = {"accuracy": lambda h: h[:, idx]}
            if r0["unseen"][idx].any():
                units["unseen_rule_accuracy"] = lambda h: h[:, idx[r0["unseen"][idx]]]
            if sel.any():
                units["paired_accuracy"] = lambda h: h[:, po[sel]] * h[:, pe[sel]]
            lv = res["levels"][f"{s}|{g}"] = {"set": s, "group": g, "n": len(idx), "models": {}}
            maj = base.get("majority_baseline", {}).get(s, {}).get("pooled" if is_pool else g)
            lv["majority_baseline"] = maj if maj is not None else _stat(bench[ref][0]["sets"][s], g, many, is_pool).get("majority_baseline")
            for name, runs in bench.items():
                stats = [_stat(x["sets"][s], g, many, is_pool) for x in runs]
                m = {k: mean_std([st[k] for st in stats]) for k in stats[0] if all(k in st for st in stats)}
                oi = [x["sets"][s].get("order_invariance", {}).get(g) for x in runs]
                if not is_pool and all(oi):
                    m["option_order_invariance"] = mean_std([o["argmax_agreement"] for o in oi])
                row = lv["models"][name] = {"metrics": m, "difference_to_" + ref: {}}
                if name != ref:
                    for metric, unit in units.items():
                        d, lo, hi = bootstrap_ci(unit(hits[name]), unit(hits[ref]), b["resamples"], b["seed"], b["ci_level"])
                        row["difference_to_" + ref][metric] = {"difference": d, "ci_low": lo, "ci_high": hi,
                                                              "excludes_0": bool(lo > 0 or hi < 0), "reliable_improvement": bool(lo > 0)}
    out_base = Path(str(ROOT / cfg["report"]) if out is None else Path(out) / "benchmark")
    out_base.parent.mkdir(parents=True, exist_ok=True)
    Path(str(out_base) + ".json").write_text(json.dumps(res, indent=1, default=float))
    Path(str(out_base) + ".txt").write_text(bench_text(res))
    return res


def bench_text(res):
    ref, models = res["reference"], res["models"]
    cell = lambda ms: "n/a" if ms is None else f"{ms[0]:.4f} +- {ms[1]:.4f}"
    out = [f"Step 8 benchmark. Mean +- std (ddof 1) over {len(next(iter(models.values()))['seeds'])} seeds. Better direction in each column.",
           f"Difference to {ref}: {res['bootstrap']['ci_level']:.0%} paired bootstrap CI ({res['bootstrap']['resamples']} resamples, "
           f"seed {res['bootstrap']['seed']}), the mean over the seed pairs; the same seed of {ref} is the pair. The CI covers the "
           "records, not the seed variance.", f"Statistically reliable (Step 0): the CI of the difference to {ref} excludes 0. "
           "'better' = CI above 0 (a reliable improvement); 'worse' = CI below 0.", f"Evaluation limits (tokens): {res['eval_limits']}",
           "", "Models: parameters, and latency on the L4 at batch size 1 (from the training run)",
           "| model | family | params_total | params_trainable | p50 ms (lower is better) | p95 ms (lower is better) |", "|---|---|---|---|---|---|"]
    for n, m in models.items():
        f = lambda v: "n/a" if v is None else f"{v:.2f}"
        out.append(f"| {n} | {m['family']} | {m['params_total']} | {m['params_trainable']} | {f(m['latency_bs1_p50_ms'])} | {f(m['latency_bs1_p95_ms'])} |")
    if res["baselines"].get("fact_count"):
        out += ["", "Fact-count probe (trained on train; accuracy, higher is better): " + json.dumps(res["baselines"]["fact_count"])]
    compact = [lv for lv in res["levels"].values() if lv["set"] in ("boardgameqa", "supernat") and lv["group"] != "pooled"]
    for lv in res["levels"].values():
        if lv in compact:
            continue
        keys = [k for k in (*TABLE_KEYS, "option_order_invariance") if k in lv["models"][ref]["metrics"]]
        out += ["", f"## {lv['set']} / {lv['group']} (n = {lv['n']}); majority baseline (reference): {lv['majority_baseline']:.4f}",
                "| model | " + " | ".join(f"{k} ({'lower' if k in LOWER else 'higher'} is better)" for k in keys) + " |", "|---|" + "---|" * len(keys)]
        out += [f"| {n} | " + " | ".join(cell(row["metrics"].get(k)) for k in keys) + " |" for n, row in lv["models"].items()]
        out += ["", f"Difference to {ref} (higher is better)", "| model | metric | difference | CI low | CI high | CI excludes 0 | verdict |", "|---|---|---|---|---|---|---|"]
        out += [f"| {n} | {k} | {v['difference']:+.4f} | {v['ci_low']:+.4f} | {v['ci_high']:+.4f} | {'yes' if v['excludes_0'] else 'no'} | "
                f"{'better' if v['ci_low'] > 0 else 'worse' if v['ci_high'] < 0 else 'no reliable difference'} |"
                for n, row in lv["models"].items() for k, v in row[f"difference_to_{ref}"].items()]
    for s in ("boardgameqa", "supernat"):
        lvs = [lv for lv in compact if lv["set"] == s]
        if lvs:
            names = list(models)
            out += ["", f"## {s}: accuracy per {'subset' if s == 'boardgameqa' else 'task'}, mean over seeds (higher is better); CIs in the JSON",
                    "| group | n | majority baseline | " + " | ".join(names) + " |", "|---|---|---|" + "---|" * len(names)]
            out += [f"| {lv['group']} | {lv['n']} | {lv['majority_baseline']:.3f} | " + " | ".join(f"{lv['models'][n]['metrics']['accuracy'][0]:.3f}" for n in names) + " |"
                    for lv in lvs]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?")
    ap.add_argument("--bench", metavar="CONFIG", help="Step 8: write the benchmark report of configs/step8_eval.yaml")
    ap.add_argument("--metric", default="unseen_rule_accuracy")
    ap.add_argument("--ref", default="A", help="the reference architecture of the paired bootstrap")
    args = ap.parse_args()
    if args.bench:
        bench_report(args.bench)
    else:
        aggregate(args.root, args.metric, args.ref)
