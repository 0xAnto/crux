"""Step 1: fine-tune Ettin-17M on RuleTaker. Run: python -m crux.train_step1 --config configs/step1_ruletaker.yaml"""
import argparse
import json
from pathlib import Path

import torch
import transformers
import yaml
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer,
                          TrainingArguments, set_seed)

from crux.data.ruletaker import download, load_split

ROOT = Path(__file__).resolve().parents[1]


def apply_overrides(cfg, overrides):
    """Apply "a.b=value" overrides. The value is parsed as YAML."""
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        *path, last = key.split(".")
        for p in path:
            node = node[p]
        if last not in node:
            raise KeyError(f"Unknown config key: {key}")
        node[last] = yaml.safe_load(value)
    return cfg


def encode(tokenizer, examples, max_length):
    """Sentence-pair input: [CLS] context [SEP] question [SEP]. Truncate the context only."""
    contexts = [e["context"] for e in examples]
    questions = [e["question"] for e in examples]
    full = tokenizer(contexts, questions)["input_ids"]
    n_truncated = sum(len(ids) > max_length for ids in full)
    enc = tokenizer(contexts, questions, truncation="only_first", max_length=max_length)
    rows = [{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i], "labels": e["label"]}
            for i, e in enumerate(examples)]
    return rows, n_truncated


def accuracy(eval_pred):
    logits, labels = eval_pred
    return {"accuracy": float((logits.argmax(-1) == labels).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="a.b=value", help="override config values")
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    args = ap.parse_args()

    cfg = apply_overrides(yaml.safe_load(Path(args.config).read_text()), args.set)
    protocol = yaml.safe_load((ROOT / "configs" / "protocol.yaml").read_text())
    t = cfg["train"]
    seed = protocol["seeds"][t["seed_index"]]
    set_seed(seed)

    d = cfg["data"]
    data_dir = download(d["url"], ROOT / d["cache_dir"], d["variant"], d.get("sha256"))
    labels = d["labels"]  # the number of classes comes from here
    train_ex = load_split(data_dir, d["train_split"], labels, d.get("max_train_examples"))
    eval_ex = load_split(data_dir, d["eval_split"], labels, d.get("max_eval_examples"))

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["id"])
    train_ds, train_trunc = encode(tokenizer, train_ex, cfg["max_length"])
    eval_ds, eval_trunc = encode(tokenizer, eval_ex, cfg["max_length"])

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"]["id"], num_labels=len(labels), attn_implementation=cfg["model"]["attn_implementation"],
        id2label=dict(enumerate(labels)), label2id={n: i for i, n in enumerate(labels)})
    params_total = sum(p.numel() for p in model.parameters())
    params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    out_dir = ROOT / "runs" / "step1" / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    use_cuda = torch.cuda.is_available() and not args.cpu
    resolved = {**cfg, "seed": seed, "fp16_used": bool(t["fp16"] and use_cuda), "n_train": len(train_ex),
                "n_eval": len(eval_ex), "params_total": params_total, "params_trainable": params_trainable,
                "torch": str(torch.__version__), "transformers": transformers.__version__, "cuda": use_cuda}
    (out_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))

    bs = t["per_device_train_batch_size"]
    total_steps = t["max_steps"] if t["max_steps"] > 0 else -(-len(train_ds) // bs) * t["num_train_epochs"]
    warmup_steps = int(t["warmup_ratio"] * total_steps)  # an int works in transformers 4.x and 5.x

    targs = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"), seed=seed, data_seed=seed,
        fp16=resolved["fp16_used"], use_cpu=not use_cuda and args.cpu,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        learning_rate=t["learning_rate"], weight_decay=t["weight_decay"], warmup_steps=warmup_steps,
        lr_scheduler_type=t["lr_scheduler_type"], num_train_epochs=t["num_train_epochs"], max_steps=t["max_steps"],
        eval_strategy="steps", eval_steps=t["eval_steps"], save_strategy="steps", save_steps=t["eval_steps"],
        logging_steps=t["logging_steps"], save_total_limit=t["save_total_limit"],
        load_best_model_at_end=True, metric_for_best_model="accuracy", greater_is_better=True,  # selection on dev
        dataloader_num_workers=t["dataloader_num_workers"], report_to="none")
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds,
                      data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=accuracy)
    trainer.train()
    dev = trainer.evaluate()  # best checkpoint, dev split

    trainer.save_model(str(out_dir / "best_model"))
    tokenizer.save_pretrained(str(out_dir / "best_model"))
    (out_dir / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1))
    metrics = {"split": d["eval_split"], "dev_accuracy": dev["eval_accuracy"], "dev_loss": dev["eval_loss"],
               "best_checkpoint": trainer.state.best_model_checkpoint, "global_step": trainer.state.global_step,
               "n_train": len(train_ex), "n_eval": len(eval_ex), "n_truncated_train": train_trunc,
               "n_truncated_eval": eval_trunc, "max_length": cfg["max_length"], "seed": seed,
               "params_total": params_total, "params_trainable": params_trainable}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
