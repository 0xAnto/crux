"""Step 8: one evaluation script and one protocol for all models.

Run (the test split needs --final; the final run happens one time, see steps.md Step 8):
  python -m crux.benchmark [--final] [--baselines] [--runs A/s0 D/s1 ...] [--sets validation ...] [--limit N] [--cpu]
For each run folder <root>/<model>/t0/s<k> of configs/step8_eval.yaml it builds the model from resolved_config.yaml
(else from the step config), loads best.pt, and evaluates the sets. A run folder that does not exist is skipped.
It writes <out_root>/<model>/s<k>/bench.json and probs_<set>.npz (ids, hits, probs, lens, and the arrays that
crux.aggregate needs: group, unseen, pair_orig, pair_edit). A set whose output exists is skipped, so a stopped job resumes.
Sets: validation, the evaluation-only benchmarks (boardgameqa, dynabench, supernat), and test.
- Length limits: configs/step8_eval_data.yaml (the same for all models). The tokenizer comes from the data config of each run.
- Without --final the script refuses the test split. `Data.get` is the only place that reads it.
- Option-order invariance (validation and test): crux.train.order_invariance for each source / subset.
- --baselines: the majority baselines (test: the train label as reference) and the fact-count probe, written to
  <out_root>/baselines.json. Both are the same for all models.
"""
import argparse
import gc
import json
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import transformers
import yaml

from crux.aggregate import pair_index
from crux.data.batching import make_dataloader
from crux.metrics import IDS, _leaked, evaluate, majority_baseline, seen_ids
from crux.models import build_model
from crux.splits import ROOT, in_dev, load_split, unit_key
from crux.train import input_mode, load_config, order_invariance, predict

BENCHMARKS = ("boardgameqa", "dynabench", "supernat")
WITH_ORDER = ("validation", "test")  # the sets with the option-order invariance


class Data:
    """The records of the evaluation sets. The test split is read here, one time, and only if final=True."""

    def __init__(self, final=False, limit=None):
        self.final, self.limit, self.cache = final, limit, {}

    def get(self, name):
        if name not in self.cache:
            if name == "test" and not self.final:
                raise PermissionError("the test split needs --final (Step 8 only)")
            records = load_split(name, allow_test=self.final) if name == "test" else load_split(name)
            if len({r["id"] for r in records}) != len(records):
                raise ValueError(f"{name}: record ids are not unique")
            if name == "supernat":  # one group for each task
                records = [{**r, "meta": {**r["meta"], "subset": r["meta"]["task_id"]}} for r in records]
            if self.limit and name != "train":  # a sample of whole units (pairs and paraphrases stay together)
                records = [r for r in records if in_dev(unit_key(r), 0, self.limit / len(records))]
            self.cache[name] = records
        return self.cache[name]


def group_of(r):
    """The key of `crux.metrics.evaluate` for the record: source or source/subset."""
    return r["source"] + ("/" + r["meta"]["subset"] if "subset" in r["meta"] else "")


def by_group(records):
    groups = defaultdict(list)
    for i, r in enumerate(records):
        groups[group_of(r)].append(i)
    return groups


def pooled(records):
    """The records without `meta.subset`: one value for the whole benchmark."""
    return [{**r, "meta": {k: v for k, v in r["meta"].items() if k != "subset"}} for r in records]


def majorities(records, ref=None):
    """Majority baseline (accuracy) of each group. `ref`: the records that give the majority label (train), else the
    group itself. "pooled": the mean over the groups, weighted by their size."""
    out = {g: majority_baseline([records[i] for i in ix], ref=ref)["majority_baseline"] for g, ix in by_group(records).items()}
    sizes = {g: len(ix) for g, ix in by_group(records).items()}
    out["pooled"] = sum(out[g] * sizes[g] for g in sizes) / len(records)
    return out


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    Path(tmp).write_text(json.dumps(obj, indent=1, default=float))
    os.replace(tmp, path)


def run_config(run, spec, name):
    """The config of a run: resolved_config.yaml, else the step config with the model name as architecture."""
    if (run / "resolved_config.yaml").exists():
        cfg = yaml.safe_load((run / "resolved_config.yaml").read_text())
    else:
        cfg = load_config(ROOT / spec["config"], name, 0)
    seeds = yaml.safe_load((ROOT / "configs" / "protocol.yaml").read_text())["seeds"]
    cfg.setdefault("seed", seeds[int(run.name[1:])])
    return cfg


def evaluate_set(name, model, loader, records, train, ctx):
    """Predict one set and return (the bench.json entry, the arrays of probs_<set>.npz)."""
    t0 = time.time()
    by_id = predict(model, loader, ctx["device"], ctx["fp16"])
    probs = [by_id[r["id"]] for r in records]
    entry = {"n": len(records), "limit": ctx["limit"], "truncated": dict(loader.dataset.truncated),
             "metrics": evaluate(records, probs, train=train)}
    if name in BENCHMARKS:
        entry["pooled"] = evaluate(pooled(records), probs)
    if name in WITH_ORDER:
        entry["order_invariance"] = {g: order_invariance(model, ctx["tokenizer"], [records[i] for i in ix],
                                                         [probs[i] for i in ix], train, ctx["dcfg"], ctx["mode"],
                                                         ctx["device"], ctx["fp16"], ctx["seed"])
                                     for g, ix in by_group(records).items()}
    depth5 = [i for i, r in enumerate(records) if group_of(r) == "proofwriter/depth5"]
    for d in sorted({records[i]["meta"]["depth"] for i in depth5}):  # depth5 holds 10 questions of depth 6 or 7
        ix = [i for i in depth5 if records[i]["meta"]["depth"] == d]
        entry["metrics"][f"proofwriter/depth5/depth={d}/accuracy"] = float(np.mean([probs[i].argmax() == records[i]["target"] for i in ix]))
        entry["metrics"][f"proofwriter/depth5/depth={d}/n"] = len(ix)
    entry["seconds"] = time.time() - t0
    have = [all(k in r["meta"] for k in IDS) for r in records]
    seen = seen_ids(train) if train and any(have) else None
    gold = [int(np.argmax(r["target"])) if r["task_type"] == "scoring" else r["target"] for r in records]
    o, e = pair_index(records)
    arrays = {"ids": np.array([r["id"] for r in records]), "hits": np.array([p.argmax() == g for p, g in zip(probs, gold)]),
              "probs": np.concatenate(probs).astype(np.float32), "lens": np.array([len(p) for p in probs]),
              "group": np.array([group_of(r) for r in records]),
              "unseen": np.array([seen is not None and h and not _leaked(r, seen) for r, h in zip(records, have)]), "pair_orig": np.array(o, dtype=int), "pair_edit": np.array(e, dtype=int)}
    return entry, arrays


def bench_run(name, k, spec, args, data, device, loaders, tokenizers):
    """Evaluate the missing sets of one run. Return False if the run folder does not exist."""
    run = ROOT / spec["root"] / name / "t0" / f"s{k}"
    if not (run / "best.pt").exists():
        return False
    out = Path(args.out) / name / f"s{k}"
    bench = json.loads((out / "bench.json").read_text()) if (out / "bench.json").exists() else {"sets": {}}
    todo = [s for s in args.sets if not ((out / f"probs_{s}.npz").exists() and bench["sets"].get(s, {}).get("limit") == args.limit
                                          and s in bench["sets"])]
    if not todo:
        return True
    cfg = run_config(run, spec, name)
    run_dcfg = yaml.safe_load((ROOT / cfg["data_config"]).read_text())
    limits = yaml.safe_load((ROOT / args.data_config).read_text())
    dcfg = {**run_dcfg, **{key: v for key, v in limits.items() if key != "tokenizer"}}
    mode, fp16 = input_mode(cfg["model"]), bool(cfg["train"]["fp16"]) and device.type == "cuda"
    if run_dcfg["tokenizer"] not in tokenizers:
        tokenizers[run_dcfg["tokenizer"]] = transformers.AutoTokenizer.from_pretrained(run_dcfg["tokenizer"])
    tokenizer = tokenizers[run_dcfg["tokenizer"]]
    model = build_model(cfg["model"])
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu")["model"])
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    bench.update(model=name, seed_index=k, seed=cfg["seed"], run=str(run), device=str(device), fp16=fp16, mode=mode,
                 eval_limits={key: v for key, v in dcfg.items() if key != "tokenizer"}, tokenizer=run_dcfg["tokenizer"],
                 params_total=n_params, params_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                 torch=torch.__version__, transformers=transformers.__version__, commit=commit())
    for s in todo:
        records = data.get(s)
        key = (run_dcfg["tokenizer"], mode, s, args.limit)  # the loaders (token ids) are the same for all runs of a tokenizer
        if key not in loaders:
            loaders[key] = make_dataloader(records, tokenizer, dcfg, mode, True)
        ctx = dict(device=device, fp16=fp16, tokenizer=tokenizer, dcfg=dcfg, mode=mode, seed=cfg["seed"], limit=args.limit)
        entry, arrays = evaluate_set(s, model, loaders[key], records, data.get("train") if s in WITH_ORDER else None, ctx)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / f"probs_{s}.tmp.npz", **arrays)
        os.replace(out / f"probs_{s}.tmp.npz", out / f"probs_{s}.npz")
        bench["sets"][s] = entry
        bench["complete"] = all(x in bench["sets"] for x in args.all_sets)
        save_json(bench, out / "bench.json")  # after the probs: a set in bench.json has its probs file
        print(f"{name}/s{k} {s}: n={entry['n']} {entry['seconds']:.0f} s truncated={entry['truncated']['records']}", flush=True)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return True


def commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or None
    except OSError:
        return None


def baselines(data, out):
    """The reference values, the same for all models: the majority baselines and the fact-count probe (Step 3 code).
    Majority of validation and test: the train label. Majority of a benchmark: its own label (the label names differ
    from train). The probe trains on the orig binary records of train and scores dev_id, validation (and test if --final)."""
    sets = ["validation", *BENCHMARKS] + (["test"] if data.final else [])
    res = {"majority_baseline": {s: majorities(data.get(s), data.get("train") if s in WITH_ORDER else None) for s in sets}}
    save_json(res, out / "baselines.json")
    from crux.splits import fact_count

    names = ("train", "dev_id", "validation") + (("test",) if data.final else ())
    theories = {}
    with open(ROOT / "data" / "processed" / "synthetic" / "main" / "theories.jsonl", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            theories[o["theory_hash"]] = o["theory"]
    res["fact_count"] = fact_count({n: data.get(n) for n in names}, theories, names=names)
    save_json(res, out / "baselines.json")
    print("baselines:", json.dumps(res["fact_count"]), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/step8_eval.yaml")
    ap.add_argument("--data-config", help="the length limits (default: eval_data_config of --config)")
    ap.add_argument("--out", help="default: out_root of --config")
    ap.add_argument("--runs", nargs="*", help="model/s<k> (default: all models, all seeds)")
    ap.add_argument("--sets", nargs="*", help="default: all sets of --config, without test if --final is absent")
    ap.add_argument("--final", action="store_true", help="evaluate the test split (Step 8, one time)")
    ap.add_argument("--baselines", action="store_true", help="also compute the baselines")
    ap.add_argument("--limit", type=int, help="a sample of about N records of each set (checks only; test is refused)")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    args.data_config = args.data_config or cfg["eval_data_config"]
    args.out = args.out or ROOT / cfg["out_root"]
    args.all_sets = cfg["sets"]
    args.sets = args.sets or [s for s in cfg["sets"] if args.final or s != "test"]
    if "test" in args.sets and (not args.final or args.limit):
        raise SystemExit("the test split needs --final and no --limit")
    n_seeds = len(yaml.safe_load((ROOT / "configs" / "protocol.yaml").read_text())["seeds"])
    runs = args.runs or [f"{m}/s{k}" for m in cfg["models"] for k in range(n_seeds)]
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")  # never MPS
    data, loaders, tokenizers = Data(args.final, args.limit), {}, {}
    for r in runs:
        name, k = r.rsplit("/s", 1)
        if not bench_run(name, int(k), cfg["models"][name], args, data, device, loaders, tokenizers):
            print(f"skip {r}: no run folder", flush=True)
    done = Path(args.out) / "baselines.json"
    if args.baselines and not (done.exists() and "fact_count" in json.loads(done.read_text())):
        baselines(data, Path(args.out))


if __name__ == "__main__":
    main()
