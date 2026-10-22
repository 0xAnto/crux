"""Step 10: build the release folders for the Hugging Face Hub. Nothing is uploaded.

Run: python -m crux.release [--out release]
- model/ (the model repo 0xanto/crux): weights/<model>/s<k>/model.safetensors and config.yaml for the 11 models of
  configs/step8_eval.yaml. The file keeps the model state dict of best.pt only (no optimizer state).
  README.md comes from docs/model_card.md.
- dataset/ (the dataset repo 0xanto/crux-data): data/ with the splits that DATA_LICENSES.md allows to redistribute
  (the synthetic splits, test with the ProofWriter records, BoardgameQA, DynaBench). Super-NaturalInstructions: task
  IDs and record hashes only, no instances. README.md comes from docs/dataset_card.md.
Each folder also holds NOTICE, LICENSE, DATA_LICENSES.md, and MANIFEST.json (SHA-256 of every file).
The builder copies the test split as a file. It does not read the records.
Upload: hf upload 0xanto/crux release/model . ; hf upload 0xanto/crux-data release/dataset . --repo-type dataset
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file

from crux.benchmark import commit, run_config
from crux.splits import ROOT

FULL = ("train", "dev_id", "validation", "test", "boardgameqa", "dynabench")  # DATA_LICENSES.md: redistribution allowed
IDS_ONLY = ("supernat",)  # DATA_LICENSES.md: the task instances stay out
STATIC = ("NOTICE", "LICENSE", "DATA_LICENSES.md")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def export_weights(root, out):
    """Write model.safetensors and config.yaml for each model and seed. A run folder that does not exist is skipped."""
    models = yaml.safe_load((root / "configs" / "step8_eval.yaml").read_text())["models"]
    for name, spec in models.items():
        for run in sorted((root / spec["root"] / name / "t0").glob("s*")):
            if not (run / "best.pt").exists():
                continue
            dest = out / "weights" / name / run.name
            dest.mkdir(parents=True)
            state = torch.load(run / "best.pt", map_location="cpu", weights_only=True)["model"]
            save_file({k: v.contiguous() for k, v in state.items()}, dest / "model.safetensors", metadata={"format": "pt"})
            (dest / "config.yaml").write_text(yaml.safe_dump(run_config(run, spec, name)))


def export_data(root, out, splits):
    """Copy the files of FULL. For IDS_ONLY write the record hashes and the per-task license, never the text."""
    for name in FULL:
        src = root / splits[name]["path"]
        if sha256(src) != splits[name]["sha256"]:
            raise ValueError(f"{name}: the hash differs from configs/splits_manifest.json")
        dest = out / "data" / Path(splits[name]["path"]).relative_to("data/splits")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
    for name in IDS_ONLY:
        src = root / splits[name]["path"]
        if sha256(src) != splits[name]["sha256"]:
            raise ValueError(f"{name}: the hash differs from configs/splits_manifest.json")
        tasks, dest = {}, out / "data" / "benchmarks"
        dest.mkdir(parents=True, exist_ok=True)
        with open(src, encoding="utf-8") as f, open(dest / f"{name}_ids.jsonl", "w", encoding="utf-8") as g:
            for line in f:
                r = json.loads(line)
                tasks[r["meta"]["task_id"]] = {k: r["meta"][k] for k in ("category", "license")}
                digest = hashlib.sha256(json.dumps(r, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                g.write(json.dumps({"id": r["id"], "task_id": r["meta"]["task_id"], "sha256": digest}) + "\n")
        (dest / f"{name}_tasks.json").write_text(json.dumps(tasks, indent=1, sort_keys=True))


def build(out, root=ROOT):
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    splits = json.loads((root / "configs" / "splits_manifest.json").read_text())["splits"]
    export_weights(root, out / "model")
    export_data(root, out / "dataset", splits)
    shutil.copyfile(root / "configs" / "splits_manifest.json", out / "dataset" / "splits_manifest.json")
    manifests = {}
    for part, card in (("model", "model_card.md"), ("dataset", "dataset_card.md")):
        d = out / part
        for f in STATIC:
            shutil.copyfile(root / f, d / f)
        shutil.copyfile(root / "docs" / card, d / "README.md")
        files = {p.relative_to(d).as_posix(): sha256(p) for p in sorted(d.rglob("*")) if p.is_file()}
        manifests[part] = {"commit": commit(), "splits": {n: {"sha256": s["sha256"], "n": s["n"]} for n, s in splits.items()},
                           "files": files}
        (d / "MANIFEST.json").write_text(json.dumps(manifests[part], indent=1))
    return manifests


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "release"))
    args = ap.parse_args()
    for part, m in build(args.out).items():
        print(f"{part}: {len(m['files'])} files")


if __name__ == "__main__":
    main()
