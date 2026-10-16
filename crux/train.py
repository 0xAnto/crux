"""Step 5: train and evaluate one model (A, B, B-ablation, A-param) with one trial and one seed.

Run: python -m crux.train --config configs/step5.yaml --arch B --trial 0 --seed-index 0
The run trains exactly `budget.training_steps` steps (configs/protocol.yaml), evaluates on validation every
`eval_every` steps, and keeps the best checkpoint on validation. The run resumes from `last.pt`.
It writes to <out_root>/<arch>/t<trial>/s<seed_index>/ : resolved_config.yaml, train.log (JSON lines),
best.pt, metrics.json, val_probs.npz. It reads only the train, dev_id, and validation splits.
"""
import argparse
import json
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


@torch.no_grad()
def time_batches(model, batches, device, fp16, warmup):
    """Milliseconds per record of each batch (forward only, data already on the device, after warmup)."""
    model.eval()
    batches = [(to_device(b, device), len(b["ids"])) for b in batches]
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


def latency(model, tokenizer, records, dcfg, lcfg, device, fp16):
    """Latency at batch size 1 and with the evaluation batches (token budget sampler)."""
    one = make_dataloader(records[:lcfg["n_batch1"] + lcfg["warmup"]], tokenizer, {**dcfg, "max_batch_size": 1},
                          "joint", True)
    loader = make_dataloader(records, tokenizer, dcfg, "joint", True)
    sampler = loader.batch_sampler.batches  # spread the timed batches over the lengths
    pick = sampler[::max(1, len(sampler) // (lcfg["n_batches"] + lcfg["warmup"]))][:lcfg["n_batches"] + lcfg["warmup"]]
    batches = [loader.collate_fn([loader.dataset[i] for i in b]) for b in pick]
    return {"device": str(device), "batch_size_1": time_batches(model, list(one), device, fp16, lcfg["warmup"]),
            "eval_batches": time_batches(model, batches, device, fp16, lcfg["warmup"])}


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


def train(cfg, data, tokenizer, model, out, device, steps, seed):
    """Train `model` for `steps` steps and write the results to `out`. `data` holds the record lists
    "train", "validation", and "dev_id". Return the content of metrics.json."""
    t, dcfg = cfg["train"], cfg["data_cfg"]
    fp16 = bool(t["fp16"]) and device.type == "cuda"
    out.mkdir(parents=True, exist_ok=True)
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    loader = make_dataloader(data["train"], tokenizer, dcfg, "joint", True, train=True, seed=seed)
    val_loader = make_dataloader(data["validation"], tokenizer, dcfg, "joint", True)
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

    t0, run_loss, run_n = time.time() - spent, 0.0, 0
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
            loss = F.cross_entropy(logits.float(), b["labels"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), t["max_grad_norm"])
            scaler.step(opt), scaler.update(), sched.step()
            step += 1
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            run_loss, run_n = run_loss + loss.item(), run_n + 1
            if step % t["log_every"] == 0 or step == steps:
                emit(step=step, loss=run_loss / run_n, grad_norm=float(gnorm), lr=sched.get_last_lr()[0],
                     seconds=time.time() - t0)
                run_loss, run_n = 0.0, 0
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
    dev_loader = make_dataloader(data["dev_id"], tokenizer, dcfg, "joint", True)
    dev, _ = evaluate_split(model, dev_loader, data["dev_id"], data["train"], device, fp16)
    gold = [r["target"] for r in data["validation"]]
    hits = np.array([int(p.argmax()) == g for p, g in zip(probs, gold)])
    np.savez_compressed(out / "val_probs.npz", ids=np.array([r["id"] for r in data["validation"]]), hits=hits,
                        probs=np.concatenate(probs).astype(np.float32), lens=np.array([len(p) for p in probs]))
    metrics = {"arch": cfg["model"]["arch"], "hyper_layers": cfg["model"].get("hyper_layers", 0), "seed": seed,
               "steps": steps, "fp16": fp16, "best_step": best["step"], "train_seconds": time.time() - t0,
               "n_train": len(data["train"]), "validation": val, "dev_id": dev, **count_params(model),
               "static_target": getattr(model, "static_target", None),
               "latency": latency(model, tokenizer, data["validation"], dcfg, cfg["latency"], device, fp16)}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    (out / "last.pt").unlink()
    return metrics


def load_config(path, arch=None, trial=None, overrides=()):
    cfg = yaml.safe_load(Path(path).read_text())
    if arch:
        cfg["model"].update(cfg["architectures"][arch])
    if trial is not None:
        cfg["train"].update(cfg["search"]["trials"][trial])
    return apply_overrides(cfg, overrides)


def main():
    from transformers import AutoTokenizer

    from crux.splits import load_split

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arch", help="a key of `architectures` in the config")
    ap.add_argument("--trial", type=int, help="an index of `search.trials` in the config")
    ap.add_argument("--seed-index", type=int, default=0, help="index in `seeds` of configs/protocol.yaml")
    ap.add_argument("--out", help="run folder (default: <out_root>/<arch>/t<trial>/s<seed_index>)")
    ap.add_argument("--set", nargs="*", default=[], metavar="a.b=value", help="override config values")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

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
    cap = cfg["data"]  # max_train and max_eval are for smoke runs only
    data = {k: load_split(k)[:cap[c]] for k, c in (("train", "max_train"), ("validation", "max_eval"),
                                                    ("dev_id", "max_eval"))}
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
