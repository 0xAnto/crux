"""The release builder on a tiny fake repository: the manifest hashes are right, and no Super-NaturalInstructions text leaks."""
import hashlib
import json

import torch
import yaml
from safetensors.torch import load_file

from crux.release import build

SNI_TEXT = ("SECRET_STATE_TEXT", "SECRET_INSTRUCTION_TEXT", "SECRET_OPTION_TEXT")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_release(tmp_path):
    root, splits = tmp_path / "repo", {}
    for name in ("train", "dev_id", "validation", "test", "boardgameqa", "dynabench", "supernat"):
        path = root / "data" / "splits" / ("benchmarks" if name in ("boardgameqa", "dynabench", "supernat") else "") / f"{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"id": f"{name}-0", "source": name, "state": "s", "instruction": "i", "options": ["a"], "target": 0, "meta": {}}
        if name == "test":  # two sources in two blocks, as in the real test split
            rec = [dict(rec, id=f"t{i}", source=s) for i, s in enumerate(("proofwriter", "proofwriter", "synthetic"))]
        if name == "supernat":
            rec |= {"state": SNI_TEXT[0], "instruction": SNI_TEXT[1], "options": [SNI_TEXT[2]],
                    "meta": {"task_id": "task001_x", "category": "Cat", "license": "MIT", "label": SNI_TEXT[2]}}
        path.write_text("".join(json.dumps(r) + "\n" for r in (rec if isinstance(rec, list) else [rec])))
        splits[name] = {"path": path.relative_to(root).as_posix(), "sha256": sha(path), "n": 1}
    (root / "configs").mkdir()
    (root / "configs" / "splits_manifest.json").write_text(json.dumps({"splits": splits}))
    (root / "configs" / "step8_eval.yaml").write_text(yaml.safe_dump({"models": {"M": {"root": "runs/x", "config": "none"}}}))
    run = root / "runs" / "x" / "M" / "t0" / "s0"
    run.mkdir(parents=True)
    torch.save({"step": 1, "model": {"w": torch.ones(2, 2)}, "optimizer": {"big": torch.zeros(9)}}, run / "best.pt")
    (run / "resolved_config.yaml").write_text(yaml.safe_dump({"model": {"arch": "A"}}))
    (root / "docs").mkdir()
    (root / "docs" / "model_card.md").write_text("card")
    (root / "docs" / "dataset_card.md").write_text("card")
    for f in ("NOTICE", "LICENSE", "DATA_LICENSES.md"):
        (root / f).write_text(f)

    out = tmp_path / "release"
    manifests = build(out, root)

    assert manifests["dataset"]["splits"]["train"]["sha256"] == splits["train"]["sha256"]
    assert "weights/M/s0/model.safetensors" in manifests["model"]["files"]
    assert list(load_file(out / "model/weights/M/s0/model.safetensors")) == ["w"]  # no optimizer state
    for part, manifest in manifests.items():
        assert "README.md" in manifest["files"] and "NOTICE" in manifest["files"]
        for rel, digest in manifest["files"].items():
            assert sha(out / part / rel) == digest
    assert (out / "dataset" / "data" / "benchmarks" / "supernat_ids.jsonl").exists()
    order = manifests["dataset"]["test_files_in_order"]
    assert order == ["data/test_proofwriter.jsonl", "data/test_synthetic.jsonl"]
    assert hashlib.sha256(b"".join((out / "dataset" / f).read_bytes() for f in order)).hexdigest() == splits["test"]["sha256"]
    for p in out.rglob("*"):
        if p.is_file():
            assert not any(t.encode() in p.read_bytes() for t in SNI_TEXT), p
