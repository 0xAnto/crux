"""crux.benchmark and the Step 8 report: the test-split guard, the resume, the metrics, and the aggregation."""
import json
import sys

import numpy as np
import pytest
import torch
import yaml
from transformers import ModernBertConfig, ModernBertModel

import crux.benchmark as B
from crux.aggregate import bench_report, bootstrap_ci
from crux.data.batching import make_dataloader
from crux.models import Scorer
from crux.train import evaluate_split

CLS, SEP, PAD = 1, 2, 0
OPTIONS = ["True", "False", "Unknown"]


class WordTokenizer:
    """Each word is one token."""
    cls_token_id, sep_token_id, pad_token_id = CLS, SEP, PAD

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [[3 + sum(map(ord, w)) % 50 for w in t.split()] for t in texts]}


def records(n, source="synthetic", start=0):
    return [{"id": f"{source}{i}", "source": source, "task_type": "binary", "state": f"fact{i % 5} holds", "instruction":
             f"If a then b. Question: is b {i % 3}?", "options": OPTIONS, "target": (i * 7) % 3, "meta": {}}
            for i in range(start, start + n)]


def tiny_model():
    torch.manual_seed(0)
    cfg = ModernBertConfig(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=2,
                           pad_token_id=PAD, cls_token_id=CLS, bos_token_id=CLS, eos_token_id=SEP, sep_token_id=SEP,
                           attn_implementation="sdpa")
    return Scorer(ModernBertModel(cfg).eval(), "A")


class FakeData:
    def __init__(self):
        self.sets = {"train": records(8, "train"), "validation": records(12), "dynabench": records(6, "dynabench")}

    def get(self, name):
        return self.sets[name]


def test_the_test_split_needs_the_final_flag(monkeypatch):
    calls = []
    monkeypatch.setattr(B, "load_split", lambda name, allow_test=False: calls.append((name, allow_test)) or records(3))
    with pytest.raises(PermissionError):
        B.Data(final=False).get("test")
    assert calls == []
    monkeypatch.setattr(sys, "argv", ["benchmark", "--sets", "test"])  # a script call without --final
    with pytest.raises(SystemExit):
        B.main()
    assert calls == []
    data = B.Data(final=True)
    data.get("test"), data.get("test")
    assert calls == [("test", True)]  # one read for any number of calls


def test_a_run_matches_evaluate_split_and_a_second_call_skips_the_finished_sets(tmp_path, monkeypatch):
    run = tmp_path / "runs" / "M" / "t0" / "s0"
    run.mkdir(parents=True)
    model = tiny_model()
    torch.save({"model": model.state_dict()}, run / "best.pt")
    (run / "resolved_config.yaml").write_text(yaml.safe_dump({
        "model": {"arch": "A"}, "train": {"fp16": False}, "data_config": "configs/step2_data.yaml", "seed": 13}))
    monkeypatch.setattr(B, "build_model", lambda cfg: tiny_model())
    monkeypatch.setattr(B.transformers.AutoTokenizer, "from_pretrained", lambda name: WordTokenizer())
    data, args = FakeData(), type("A", (), {"out": tmp_path / "out", "data_config": "configs/step8_eval_data.yaml", "limit": None,
                                          "sets": ["dynabench", "validation"], "all_sets": ["dynabench", "validation"]})
    spec, device = {"root": str(tmp_path / "runs")}, torch.device("cpu")
    assert B.bench_run("M", 0, spec, args, data, device, {}, {}) is True
    bench = json.loads((tmp_path / "out" / "M" / "s0" / "bench.json").read_text())
    dcfg = {**yaml.safe_load(open(B.ROOT / "configs/step2_data.yaml")), **yaml.safe_load(open(B.ROOT / "configs/step8_eval_data.yaml"))}
    expected, _ = evaluate_split(model, make_dataloader(data.sets["validation"], WordTokenizer(), dcfg, "joint", True),
                                 data.sets["validation"], data.sets["train"], device, False)
    assert bench["complete"] and bench["sets"]["validation"]["metrics"] == pytest.approx(expected)
    assert bench["sets"]["validation"]["order_invariance"]["synthetic"]["argmax_agreement"] == 1.0
    assert bench["sets"]["dynabench"]["pooled"]["accuracy"] == bench["sets"]["dynabench"]["metrics"]["accuracy"]
    # a second call predicts nothing; a removed set is evaluated again, the other one is not
    monkeypatch.setattr(B, "predict", lambda *a, **k: pytest.fail("a finished set was evaluated again"))
    assert B.bench_run("M", 0, spec, args, data, device, {}, {}) is True
    (tmp_path / "out" / "M" / "s0" / "probs_dynabench.npz").unlink()
    monkeypatch.undo()
    monkeypatch.setattr(B, "build_model", lambda cfg: tiny_model())
    monkeypatch.setattr(B.transformers.AutoTokenizer, "from_pretrained", lambda name: WordTokenizer())
    seen = []
    real = B.predict
    monkeypatch.setattr(B, "predict", lambda m, loader, *a: seen.append(len(loader.dataset)) or real(m, loader, *a))
    B.bench_run("M", 0, spec, args, data, device, {}, {})
    assert seen == [6]
    assert B.bench_run("Missing", 0, spec, args, data, device, {}, {}) is False


def fake_run(out, name, seed, hits, acc):
    """A bench.json and a probs file of the set "validation": 40 records, 20 pairs, all with unseen ids."""
    d = out / name / f"s{seed}"
    d.mkdir(parents=True)
    n = len(hits)
    np.savez_compressed(d / "probs_validation.npz", ids=np.array([str(i) for i in range(n)]), hits=hits, probs=np.zeros(3 * n, "f4"),
                        lens=np.full(n, 3), group=np.array(["synthetic"] * n), unseen=np.ones(n, bool),
                        pair_orig=np.arange(0, n, 2), pair_edit=np.arange(1, n, 2))
    (d / "bench.json").write_text(json.dumps({"seed": seed, "params_total": 10, "params_trainable": 10, "eval_limits": {"max_length": 1},
                                              "sets": {"validation": {"metrics": {"accuracy": acc, "ece": 0.1, "majority_baseline": 0.3},
                                                                      "order_invariance": {"synthetic": {"argmax_agreement": 1.0}}}}}))
    (out / "train" / name / "t0" / f"s{seed}").mkdir(parents=True, exist_ok=True)
    (out / "train" / name / "t0" / f"s{seed}" / "metrics.json").write_text(
        json.dumps({"latency": {"batch_size_1": {"p50_ms_per_record": 10.0 + seed, "p95_ms_per_record": 12.0}}}))


def test_the_report_gives_mean_std_and_the_paired_bootstrap_ci_of_the_difference_to_a(tmp_path):
    rng = np.random.default_rng(0)
    base = [rng.random(40) < 0.5 for _ in range(3)]
    better = [h | (rng.random(40) < 0.3) for h in base]  # X is higher than A in every record that it changes
    for name, hits in (("A", base), ("X", better), ("Y", base)):
        for s in range(3):
            fake_run(tmp_path, name, s, hits[s], float(hits[s].mean()))
    cfg = {"out_root": "unused", "report": "unused", "sets": ["validation"], "reference": "A", "models": {
        n: {"root": str(tmp_path / "train"), "family": "A"} for n in "AXY"}}
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    res = bench_report(str(tmp_path / "cfg.yaml"), out=tmp_path)
    lv = res["levels"]["validation|synthetic"]["models"]
    accs = [h.mean() for h in base]
    assert lv["A"]["metrics"]["accuracy"] == pytest.approx([np.mean(accs), np.std(accs, ddof=1)])
    x = lv["X"]["difference_to_A"]["accuracy"]
    expected = bootstrap_ci(np.stack(better), np.stack(base), 10000, 20261014, 0.95)  # the protocol values
    assert (x["difference"], x["ci_low"], x["ci_high"]) == pytest.approx(expected) and x["difference"] > 0
    pairs = lv["X"]["difference_to_A"]["paired_accuracy"]
    assert pairs["difference"] == pytest.approx(np.mean([(b[::2] & b[1::2]).mean() - (a[::2] & a[1::2]).mean()
                                                         for a, b in zip(base, better)]))
    y = lv["Y"]["difference_to_A"]["accuracy"]
    assert y["difference"] == 0 and not y["excludes_0"] and not y["reliable_improvement"]
    assert res["models"]["A"]["latency_bs1_p50_ms"] == pytest.approx(11.0)
    assert "(higher is better)" in (tmp_path / "benchmark.txt").read_text()
