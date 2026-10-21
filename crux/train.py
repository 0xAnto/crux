"""Steps 5 to 7: train and evaluate one model (A, B, B-ablation, A-param; C, A+CF, B+CF; D and its ablations) with one
trial and one seed.

Run: python -m crux.train --config configs/step5.yaml --arch B --trial 0 --seed-index 0
The run trains exactly `budget.training_steps` steps (configs/protocol.yaml), evaluates on validation every
`eval_every` steps, and keeps the best checkpoint on validation. The run resumes from `last.pt`.
It writes to <out_root>/<arch>/t<trial>/s<seed_index>/ : resolved_config.yaml, train.log (JSON lines),
best.pt, metrics.json, val_probs.npz. It reads only the train, dev_id, and validation splits.
Step 7: the input mode follows from the model (`input_mode`). metrics.json also holds `order_invariance`, `memory`, and
for the cache variants `latency.cache_hit`. Check a finished run folder (Steps 5 to 7) without training, on the CPU:
python -m crux.train --eval-only <run_dir>   (it writes <run_dir>/order_invariance.json)
"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from crux.data.batching import make_dataloader
from crux.metrics import evaluate
from crux.models import build_model, count_params
from crux.train_step1 import apply_overrides

ROOT = Path(__file__).resolve().parents[1]
LOGGED = ("accuracy", "unseen_rule_accuracy", "paired_accuracy", "ece")


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}


def autocast(device, fp16):
    return torch.autocast(device.type, dtype=torch.float16, enabled=fp16)


def save(obj, path):
    torch.save(obj, str(path) + ".tmp")
    os.replace(str(path) + ".tmp", path)


@torch.no_grad()
def predict(model, loader, device, fp16):
    """Return {record id: probabilities over the real options}."""
    model.eval()
    out = {}
    for batch in loader:
        with autocast(device, fp16):
            probs = model(**to_device(batch, device)).float().softmax(-1).cpu().numpy()
        for i, (rid, k) in enumerate(zip(batch["ids"], batch["option_mask"].sum(1).tolist())):
            out[rid] = probs[i, :k]
    return out


def evaluate_split(model, loader, records, train, device, fp16):
    by_id = predict(model, loader, device, fp16)
    probs = [by_id[r["id"]] for r in records]
    return evaluate(records, probs, train=train), probs


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def input_mode(model_cfg):
    """The split encoding (joint_layers > 0) reads separate sequences. All other models read joint sequences."""
    return "separate" if model_cfg.get("joint_layers", 0) else "joint"


@torch.no_grad()
def time_batches(model, batches, device, fp16, warmup, cached=()):
    """Milliseconds per record of each batch (forward only, data already on the device, after warmup).

    cached: names ("state", "instruction") whose bottom-layer states are computed before the timer starts (a cache hit)."""
    model.eval()
    prepared = []
    for batch in batches:
        b = to_device(batch, device)
        with autocast(device, fp16):
            b.update({f"{n}_cache": model.encode_bottom(b[f"{n}_ids"], b[f"{n}_mask"]) for n in cached})
        prepared.append((b, len(batch["ids"])))
    batches = prepared
    ms, warmup = [], min(warmup, len(batches) - 1)  # at least one timed batch
    for i, (b, n) in enumerate(batches):
        for _ in range(2 if i < warmup else 1):  # the first batches run twice and are not timed
            sync(device)
            t = time.perf_counter()
            with autocast(device, fp16):
                model(**b)
            sync(device)
        if i >= warmup:
            ms.append((time.perf_counter() - t) * 1000 / n)
    return {"p50_ms_per_record": float(np.percentile(ms, 50)), "p95_ms_per_record": float(np.percentile(ms, 95)),
            "n_batches": len(ms)}


@torch.no_grad()
def cuda_peak(model, batches, device, fp16):
    """Peak CUDA memory in bytes above the memory in use before the forward passes, or None off CUDA."""
    if device.type != "cuda":
        return None
    model.eval()
    batches = [to_device(b, device) for b in batches]
    torch.cuda.synchronize(device), torch.cuda.reset_peak_memory_stats(device)
    base = torch.cuda.memory_allocated(device)
    for b in batches:
        with autocast(device, fp16):
            model(**b)
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) - base


def cost_batches(loader, tokenizer, records, dcfg, lcfg, mode):
    """The batches for the cost measurements: batch size 1, and evaluation batches spread over the lengths."""
    one = make_dataloader(records[:lcfg["n_batch1"] + lcfg["warmup"]], tokenizer, {**dcfg, "max_batch_size": 1}, mode, True)
    sampler = loader.batch_sampler.batches
    pick = sampler[::max(1, len(sampler) // (lcfg["n_batches"] + lcfg["warmup"]))][:lcfg["n_batches"] + lcfg["warmup"]]
    return list(one), [loader.collate_fn([loader.dataset[i] for i in b]) for b in pick]


def latency(model, one, batches, device, fp16, warmup):
    """Latency at batch size 1 and with the evaluation batches (token budget sampler). A cache variant also gives
    `cache_hit`: the bottom-layer states of the state (and of the instruction) are ready before the timer starts."""
    out = {"device": str(device), "batch_size_1": time_batches(model, one, device, fp16, warmup),
           "eval_batches": time_batches(model, batches, device, fp16, warmup)}
    if model.J:
        out["cache_hit"] = {name: {"batch_size_1": time_batches(model, one, device, fp16, warmup, names),
                                   "eval_batches": time_batches(model, batches, device, fp16, warmup, names)}
                            for name, names in (("state", ("state",)), ("state_and_instruction", ("state", "instruction")))}
    return out


def memory(model, dataset, one=None, batches=None, device=None, fp16=False):
    """Memory of the model. model_bytes: the parameters in fp32 and fp16. cuda_peak_extra_bytes: the peak CUDA memory
    of a forward pass above the memory in use before it (null off CUDA, or without batches). cache_bytes_fp16
    (split encoding only): the mean bottom-layer states of a distinct state and of a distinct instruction of
    `dataset` (real tokens x hidden size x 2 bytes)."""
    n = sum(p.numel() for p in model.parameters())
    out = {"model_bytes": {"fp32": 4 * n, "fp16": 2 * n}, "cuda_peak_extra_bytes": {
        "batch_size_1": cuda_peak(model, one, device, fp16) if one else None,
        "eval_batches": cuda_peak(model, batches, device, fp16) if batches else None}, "cache_bytes_fp16": None}
    if model.J:
        d = model.encoder.config.hidden_size
        sizes = {k: {tuple(it[f"{k}_ids"]): len(it[f"{k}_ids"]) * d * 2 for it in dataset.items}
                 for k in ("state", "instruction")}
        out["cache_bytes_fp16"] = {f"{k}_mean": float(np.mean(list(v.values()))) for k, v in sizes.items()}
        out["cache_bytes_fp16"].update({f"n_{k}s": len(v) for k, v in sizes.items()})
    return out


def shuffle_options(records, seed):
    """Copies of the records with the options of each record in a fixed random order (the target follows).
    Return (records, perms): perms[i][j] is the canonical index of the option at position j of record i."""
    rng, out, perms = random.Random(seed), [], []
    for r in records:
        perm = list(range(len(r["options"])))
        rng.shuffle(perm)
        target = [r["target"][p] for p in perm] if r["task_type"] == "scoring" else perm.index(r["target"])
        out.append({**r, "options": [r["options"][p] for p in perm], "target": target})
        perms.append(perm)
    return out, perms


def order_invariance(model, tokenizer, records, probs, train, dcfg, mode, device, fp16, seed):
    """Option-order invariance on `records`: shuffle the options of each record (fixed seed), predict, and map the
    probabilities back to the canonical order. `probs` are the predictions in the canonical order.
    argmax_agreement: share of records with the same predicted option (higher is better). mean_abs_prob_diff and
    max_abs_prob_diff: the absolute difference of the probabilities, over all (record, option) pairs (lower is
    better). shuffled: accuracy and unseen_rule_accuracy with the shuffled order."""
    shuffled, perms = shuffle_options(records, seed)
    loader = make_dataloader(shuffled, tokenizer, dcfg, mode, True)
    by_id = predict(model, loader, device, fp16)
    back = []
    for r, perm in zip(records, perms):
        p = np.empty(len(perm))
        p[perm] = by_id[r["id"]]  # position j holds the canonical option perm[j]
        back.append(p)
    diff = np.concatenate([np.abs(p - q) for p, q in zip(probs, back)])
    m = evaluate(shuffled, [by_id[r["id"]] for r in records], train=train)
    return {"seed": seed, "n_records": len(records), "n_moved": sum(p != sorted(p) for p in perms),
            "argmax_agreement": float(np.mean([p.argmax() == q.argmax() for p, q in zip(probs, back)])),
            "mean_abs_prob_diff": float(diff.mean()), "max_abs_prob_diff": float(diff.max()),
            "shuffled": {k: m[k] for k in ("accuracy", "unseen_rule_accuracy") if k in m}}


def make_optimizer(model, t):
    pre = {id(p) for p in model.pretrained_params()}
    groups = []
    for is_pre, lr in ((True, t["lr"]), (False, t["lr"] * t["new_lr_mult"])):
        ps = [p for p in model.parameters() if p.requires_grad and (id(p) in pre) == is_pre]
        for decay in (True, False):
            sub = [p for p in ps if (p.ndim >= 2) == decay]
            if sub:
                groups.append({"params": sub, "lr": lr, "weight_decay": t["weight_decay"] if decay else 0.0})
    return torch.optim.AdamW(groups)


def consistency_loss(logits, option_mask, groups):
    """Step 6: the mean Jensen-Shannon divergence of each paraphrase group to the mean distribution of the group.

    groups: lists of batch rows with the same `meta.paraphrase_group` (2 or more rows). The rows of a group have
    the same options, so they have the same real options. Return 0 if no group is in the batch.
    """
    terms = []
    for g in groups:
        k = int(option_mask[g[0]].sum())
        logp = logits[g, :k].float().log_softmax(-1)
        logm = logp.logsumexp(0) - math.log(len(g))  # log of the mean distribution, finite when all p underflow
        terms.append((logp.exp() * (logp - logm)).sum(-1).mean())
    return torch.stack(terms).mean() if terms else logits.new_zeros(())


def paraphrase_rows(ids, group_of):
    """Batch rows of each paraphrase group with 2 or more rows in this batch."""
    rows = {}
    for i, rid in enumerate(ids):
        rows.setdefault(group_of.get(rid), []).append(i)
    return [g for k, g in rows.items() if k is not None and len(g) > 1]


def train(cfg, data, tokenizer, model, out, device, steps, seed):
    """Train `model` for `steps` steps and write the results to `out`. `data` holds the record lists
    "train", "validation", and "dev_id". Return the content of metrics.json.
    Step 6: train.paired gives unit-paired batches, and train.consistency_weight > 0 adds the consistency loss.
    Step 7: the model config sets the input mode (`input_mode`), and metrics.json holds the order invariance and the memory."""
    t, dcfg, mode = cfg["train"], cfg["data_cfg"], input_mode(cfg["model"])
    fp16 = bool(t["fp16"]) and device.type == "cuda"
    out.mkdir(parents=True, exist_ok=True)
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    loader = make_dataloader(data["train"], tokenizer, dcfg, mode, True, train=True, seed=seed,
                             paired=t.get("paired", False))
    lam = t.get("consistency_weight", 0.0)
    group_of = {r["id"]: r["meta"].get("paraphrase_group") for r in data["train"]} if lam else {}
    val_loader = make_dataloader(data["validation"], tokenizer, dcfg, mode, True)
    model.to(device)
    opt = make_optimizer(model, t)
    warm = int(t["warmup_ratio"] * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(warm, 1) if s < warm else max(0.0, (steps - s) / max(steps - warm, 1)))
    scaler = torch.amp.GradScaler(enabled=fp16)
    step, best, spent = 0, {"metric": -1.0, "step": 0}, 0.0
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device)
        model.load_state_dict(ck["model"]), opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]), scaler.load_state_dict(ck["scaler"])
        step, best, spent = ck["step"], ck["best"], ck["spent"]
        print(f"resume {out} at step {step}")
    log = open(out / "train.log", "a")

    def emit(**kw):
        log.write(json.dumps(kw) + "\n"), log.flush()

    t0, run_loss, run_cons, run_n = time.time() - spent, 0.0, 0.0, 0
    while step < steps:
        epoch, skip = divmod(step, len(loader))
        loader.batch_sampler.set_epoch(epoch)
        batches = list(loader.batch_sampler)[skip:]  # resume inside an epoch with the same order
        for batch in DataLoader(loader.dataset, batch_sampler=batches, collate_fn=loader.collate_fn,
                                num_workers=t["num_workers"]):
            model.train()
            b = to_device(batch, device)
            with autocast(device, fp16):
                logits = model(**b)
            loss = ce = F.cross_entropy(logits.float(), b["labels"])
            if lam:
                cons = consistency_loss(logits, batch["option_mask"], paraphrase_rows(batch["ids"], group_of))  # CPU mask: no sync
                loss = ce + lam * cons
                run_cons += cons.item()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), t["max_grad_norm"])
            scaler.step(opt), scaler.update(), sched.step()
            step += 1
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            run_loss, run_n = run_loss + ce.item(), run_n + 1
            if step % t["log_every"] == 0 or step == steps:
                emit(step=step, loss=run_loss / run_n, grad_norm=float(gnorm), lr=sched.get_last_lr()[0],
                     seconds=time.time() - t0, **({"consistency": run_cons / run_n} if lam else {}))
                run_loss, run_cons, run_n = 0.0, 0.0, 0
            if step % t["eval_every"] == 0 or step == steps:
                m, _ = evaluate_split(model, val_loader, data["validation"], data["train"], device, fp16)
                emit(step=step, validation={k: m[k] for k in LOGGED if k in m})
                if m[t["selection_metric"]] > best["metric"]:
                    best = {"metric": m[t["selection_metric"]], "step": step}
                    save({"step": step, "model": model.state_dict()}, out / "best.pt")
                save({"step": step, "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                      "scaler": scaler.state_dict(), "best": best, "spent": time.time() - t0}, out / "last.pt")
            if step >= steps:
                break

    # The best checkpoint on validation gives all final numbers.
    model.load_state_dict(torch.load(out / "best.pt", map_location=device)["model"])
    val, probs = evaluate_split(model, val_loader, data["validation"], data["train"], device, fp16)
    dev_loader = make_dataloader(data["dev_id"], tokenizer, dcfg, mode, True)
    dev, _ = evaluate_split(model, dev_loader, data["dev_id"], data["train"], device, fp16)
    gold = [r["target"] for r in data["validation"]]
    hits = np.array([int(p.argmax()) == g for p, g in zip(probs, gold)])
    np.savez_compressed(out / "val_probs.npz", ids=np.array([r["id"] for r in data["validation"]]), hits=hits,
                        probs=np.concatenate(probs).astype(np.float32), lens=np.array([len(p) for p in probs]))
    one, many = cost_batches(val_loader, tokenizer, data["validation"], dcfg, cfg["latency"], mode)
    metrics = {"arch": cfg["model"]["arch"], "hyper_layers": cfg["model"].get("hyper_layers", 0),
               "joint_layers": model.J, "option_attention": model.option_attn is not None, "seed": seed,
               "steps": steps, "fp16": fp16, "paired": t.get("paired", False), "consistency_weight": lam,
               "best_step": best["step"], "train_seconds": time.time() - t0,
               "n_train": len(data["train"]), "validation": val, "dev_id": dev, **count_params(model),
               "static_target": getattr(model, "static_target", None),
               "latency": latency(model, one, many, device, fp16, cfg["latency"]["warmup"]),
               "memory": memory(model, val_loader.dataset, one, many, device, fp16),
               "order_invariance": order_invariance(model, tokenizer, data["validation"], probs, data["train"], dcfg, mode,
                                                    device, fp16, seed)}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    (out / "last.pt").unlink()
    return metrics


def load_config(path, arch=None, trial=None, overrides=()):
    cfg = yaml.safe_load(Path(path).read_text())
    if arch:  # an architecture sets model keys, and with its `train` key also train keys (Step 6)
        a = dict(cfg["architectures"][arch])
        cfg["train"].update(a.pop("train", {}))
        cfg["model"].update(a)
    if trial is not None:
        cfg["train"].update(cfg["search"]["trials"][trial])
    return apply_overrides(cfg, overrides)


def load_data(cap, names):
    """The splits `names` as {name: records}. cap maps the keys max_train and max_eval to a limit (smoke runs only)."""
    from crux.splits import load_split
    return {k: load_split(k)[:cap[{"train": "max_train"}.get(k, "max_eval")]] for k in names}


def eval_only(run, cpu, config=None):
    """Check a finished run folder (Steps 5 to 7) without training: load best.pt with resolved_config.yaml, predict
    the validation split, and write order_invariance.json (order invariance and memory) into the folder.
    The runs of Steps 5 and 6 have no resolved_config.yaml: pass the config of their step, and the folder name
    <arch>/t<trial>/s<seed_index> gives the rest.
    The device is the CUDA GPU if there is one, else the CPU (never MPS). The prediction runs in fp16 on CUDA only."""
    from transformers import AutoTokenizer
    run = run.absolute()
    if (run / "resolved_config.yaml").exists():
        cfg = yaml.safe_load((run / "resolved_config.yaml").read_text())
    elif config:
        cfg = load_config(config, run.parents[1].name, int(run.parent.name[1:]))
        cfg["seed"] = yaml.safe_load((ROOT / "configs" / "protocol.yaml").read_text())["seeds"][int(run.name[1:])]
    else:
        raise SystemExit(f"{run} has no resolved_config.yaml. Pass --config.")
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    fp16 = bool(cfg["train"]["fp16"]) and device.type == "cuda"
    dcfg, mode = yaml.safe_load((ROOT / cfg["data_config"]).read_text()), input_mode(cfg["model"])
    data = load_data(cfg["data"], ("train", "validation"))
    tokenizer = AutoTokenizer.from_pretrained(dcfg["tokenizer"])
    model = build_model(cfg["model"])
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu")["model"])
    model.to(device)
    loader = make_dataloader(data["validation"], tokenizer, dcfg, mode, True)
    _, probs = evaluate_split(model, loader, data["validation"], data["train"], device, fp16)
    one, many = cost_batches(loader, tokenizer, data["validation"], dcfg, cfg["latency"], mode)
    out = {"run": str(run), "device": str(device), "fp16": fp16, "mode": mode,
           "order_invariance": order_invariance(model, tokenizer, data["validation"], probs, data["train"], dcfg, mode,
                                                device, fp16, cfg["seed"]),
           "memory": memory(model, loader.dataset, one, many, device, fp16)}
    (run / "order_invariance.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out["order_invariance"]))


def main():
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="the config file (with --eval-only: only for a run folder without resolved_config.yaml)")
    ap.add_argument("--arch", help="a key of `architectures` in the config")
    ap.add_argument("--trial", type=int, help="an index of `search.trials` in the config")
    ap.add_argument("--seed-index", type=int, default=0, help="index in `seeds` of configs/protocol.yaml")
    ap.add_argument("--out", help="run folder (default: <out_root>/<arch>/t<trial>/s<seed_index>)")
    ap.add_argument("--set", nargs="*", default=[], metavar="a.b=value", help="override config values")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--eval-only", metavar="RUN_DIR", help="no training: write order_invariance.json into a finished run folder")
    args = ap.parse_args()
    if args.eval_only:
        return eval_only(Path(args.eval_only), args.cpu, args.config)
    if not args.config:
        ap.error("--config is required")

    cfg = load_config(args.config, args.arch, args.trial, args.set)
    protocol = yaml.safe_load((ROOT / "configs" / "protocol.yaml").read_text())
    steps = cfg["train"]["steps"] or protocol["budget"]["training_steps"]  # train.steps is for smoke runs only
    if steps is None:
        raise SystemExit("budget.training_steps in configs/protocol.yaml is null. Set it before you train.")
    seed = protocol["seeds"][args.seed_index]
    out = Path(args.out) if args.out else ROOT / cfg["out_root"] / args.arch / f"t{args.trial}" / f"s{args.seed_index}"
    if (out / "metrics.json").exists():
        return print(f"done: {out}")
    device = torch.device("cpu" if args.cpu else "cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    cfg["data_cfg"] = yaml.safe_load((ROOT / cfg["data_config"]).read_text())
    data = load_data(cfg["data"], ("train", "validation", "dev_id"))
    tokenizer = AutoTokenizer.from_pretrained(cfg["data_cfg"]["tokenizer"])
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    model = build_model(cfg["model"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "resolved_config.yaml").write_text(yaml.safe_dump({**{k: v for k, v in cfg.items() if k != "data_cfg"},
                                                              "seed": seed, "steps": steps, "device": str(device)}))
    metrics = train(cfg, data, tokenizer, model, out, device, steps, seed)
    print(json.dumps({k: metrics[k] for k in ("arch", "seed", "best_step", "params_total", "params_trainable")}),
          {k: metrics["validation"][k] for k in LOGGED if k in metrics["validation"]})


if __name__ == "__main__":
    main()
