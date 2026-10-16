import copy
import json

import numpy as np
import pytest
import torch
from transformers import ModernBertConfig, ModernBertModel

from crux.aggregate import bootstrap_ci
from crux.models import Hypernet, Lora, Scorer, build_model, count_params, span_mean, static_ranks
from crux.train import train

CLS, SEP, PAD = 1, 2, 0


def tiny_encoder(layers=3):
    torch.manual_seed(0)
    cfg = ModernBertConfig(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=layers,
                           num_attention_heads=2, global_attn_every_n_layers=2, local_attention=8, pad_token_id=PAD,
                           cls_token_id=CLS, bos_token_id=CLS, eos_token_id=SEP, sep_token_id=SEP,
                           attn_implementation="sdpa")
    return ModernBertModel(cfg).eval()


def batch(instr=(10, 11, 12), n_opts=(3, 2)):
    """[CLS] state [SEP] instruction [SEP] option [SEP], padded in the length and in the options."""
    k, rows = max(n_opts), []
    for b, n in enumerate(n_opts):
        opts = []
        for o in range(n):
            s = [CLS, 20 + b, 21, SEP, *instr, SEP, 30 + o, SEP] + [PAD] * o  # option sequences differ in length
            opts.append(s)
        rows.append(opts)
    width = max(len(s) for r in rows for s in r)
    ids = torch.full((len(rows), k, width), PAD)
    mask = torch.zeros_like(ids)
    om = torch.zeros(len(rows), k, dtype=torch.bool)
    for b, r in enumerate(rows):
        for o, s in enumerate(r):
            ids[b, o, :len(s)] = torch.tensor(s)
            mask[b, o, :len([t for t in s if t != PAD])] = 1
            om[b, o] = True
    return {"input_ids": ids, "attention_mask": mask, "option_mask": om}


def twin(enc, K):
    """A and B on the same encoder weights, with the same head. B wraps linears, so only the head is copied."""
    a = Scorer(copy.deepcopy(enc), "A").eval()
    b = Scorer(copy.deepcopy(enc), "B", hyper_layers=K).eval()
    head = {k: v for k, v in a.state_dict().items() if not k.startswith("encoder.")}
    missing = b.load_state_dict(head, strict=False)
    assert all(k.startswith(("hyper.", "encoder.")) for k in missing.missing_keys) and not missing.unexpected_keys
    return a, b


def randomize_hyper(model):
    torch.manual_seed(1)
    for p in model.hyper.parameters():
        p.data.normal_(std=0.3)


@pytest.mark.parametrize("K", [0, 2])
def test_b_with_zero_delta_equals_a(K):
    a, b = twin(tiny_encoder(), K)
    x = batch()
    with torch.no_grad():
        assert torch.allclose(a(**x), b(**x), atol=1e-6)
        randomize_hyper(b)  # the same test fails with a non-zero delta: the check is not vacuous
        assert not torch.allclose(a(**x), b(**x), atol=1e-4)


def test_hypernet_output_depends_on_the_instruction():
    b = Scorer(tiny_encoder(), "B", hyper_layers=1).eval()
    randomize_hyper(b)
    seen = []
    b.hyper.register_forward_hook(lambda m, i, out: seen.append(torch.cat([a.flatten() for pair in out for a in pair])))
    with torch.no_grad():
        b(**batch(instr=(10, 11, 12), n_opts=(2,)))
        b(**batch(instr=(40, 41, 12), n_opts=(2,)))
        b(**batch(instr=(10, 11, 12), n_opts=(2,)))
    assert not torch.allclose(seen[0], seen[1], atol=1e-5) and torch.allclose(seen[0], seen[2], atol=1e-6)


def test_padded_options_get_no_probability_and_do_not_change_the_rest():
    for arch, K in (("A", 0), ("B", 0), ("B", 1)):
        m = Scorer(tiny_encoder(), arch, hyper_layers=K).eval()
        if arch == "B":
            randomize_hyper(m)
        x = batch(n_opts=(3, 2))
        with torch.no_grad():
            p = m(**x).softmax(-1)
            assert p[1, 2] == 0 and torch.allclose(p.sum(-1), torch.ones(2))
            x["input_ids"][1, 2] = 55  # garbage in the padded option
            alone = m(**batch(n_opts=(3, 2)))[1, :2].softmax(-1)
            assert torch.allclose(m(**x).softmax(-1)[1, :2], alone, atol=1e-6)
            assert torch.isfinite(m(**x)[x["option_mask"]]).all()


def test_a_param_matches_the_trainable_parameters_of_b():
    enc = tiny_encoder()
    a = count_params(Scorer(copy.deepcopy(enc), "A"))
    for K in (0, 2):
        b = count_params(Scorer(copy.deepcopy(enc), "B", hyper_layers=K))
        ap = Scorer(copy.deepcopy(enc), "A-param", hyper_layers=K)
        c = count_params(ap)
        assert b["params_trainable"] - a["params_trainable"] == b["params_extra"] > 0
        assert c["params_trainable"] - a["params_trainable"] == c["params_extra"]
        # The gap is smaller than the smallest LoRA module (rank 1 of a 32 x 32 layer = 64).
        assert 0 <= b["params_extra"] - c["params_extra"] < 64


def test_static_ranks_are_closest_below_the_budget():
    costs = [512, 640, 1024]
    used = sum(r * c for r, c in zip(static_ranks(costs, 10_000), costs))
    assert 10_000 - 512 < used <= 10_000


def test_adapters_change_only_the_top_k_layers_and_z_comes_from_below():
    a, b = twin(tiny_encoder(layers=3), 1)
    randomize_hyper(b)
    outs = {"a": [], "b": []}
    for name, m in (("a", a), ("b", b)):
        for l in m.encoder.layers:
            l.register_forward_hook(lambda mod, i, out, name=name: outs[name].append(out.detach()))
    z_in = []
    b.hyper.register_forward_pre_hook(lambda m, args: z_in.append(args[0]))
    x = batch(n_opts=(2,))
    with torch.no_grad():
        a(**x), b(**x)
    assert all(torch.equal(u, v) for u, v in zip(outs["a"][:2], outs["b"][:2]))  # layers 0, 1: no adapter
    assert not torch.allclose(outs["a"][2], outs["b"][2], atol=1e-5)  # layer 2: adapter
    ids = x["input_ids"][x["option_mask"]]
    mask = x["attention_mask"][x["option_mask"]]
    expect = span_mean(outs["b"][1], ids, mask, SEP).mean(0, keepdim=True)  # the output of layer L-K-1
    assert torch.allclose(z_in[0], expect, atol=1e-6)
    assert not any(isinstance(m, Lora) and m.delta is not None for m in b.modules())  # cleared after the pass
    assert {n for n, m in b.named_modules() if isinstance(m, Lora)} == {
        "dense", "encoder.layers.2.attn.Wqkv", "encoder.layers.2.attn.Wo"}


def test_span_mean_uses_the_tokens_between_the_first_two_seps():
    ids = torch.tensor([[CLS, 5, SEP, 6, 8, SEP, 9, SEP, PAD]])
    h = torch.arange(9.0).view(1, 9, 1)
    assert span_mean(h, ids, (ids != PAD).long(), SEP).item() == pytest.approx((3 + 4) / 2)


def test_hypernet_starts_at_zero_delta_and_the_b_factor_gets_gradient():
    h = Hypernet(8, [(8, 8)], 2, 4)
    (a, b), = h(torch.randn(3, 8))
    assert (b == 0).all() and a.abs().sum() > 0
    out = h(torch.randn(3, 8))[0]
    (torch.randn(3, 8, 8).bmm(out[0]).bmm(out[1])).sum().backward()
    assert h.fc2.weight.grad is not None and h.fc2.weight.grad.abs().sum() > 0


def tiny_records(n, offset=0):
    recs = []
    for i in range(n):
        k = 2 + i % 3
        recs.append({"id": f"r{offset + i}", "source": "t", "task_type": "multi_choice", "state": f"s{i % 5} s1",
                     "instruction": f"i{i % 4} i1 Question: q{i % 3}", "options": [f"o{j}" for j in range(k)],
                     "target": i % k, "meta": {}})
    return recs


class WordTokenizer:
    cls_token_id, sep_token_id, pad_token_id = CLS, SEP, PAD

    def __init__(self):
        self.vocab = {}

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [[self.vocab.setdefault(w, len(self.vocab) + 10) for w in t.split()] for t in texts]}


TCFG = {"data_cfg": {"max_length": 40, "max_option_length": 4, "train_batch_size": 4, "max_tokens": 2000,
                     "max_batch_size": 8},
        "train": {"lr": 1e-3, "new_lr_mult": 3, "weight_decay": 0.01, "warmup_ratio": 0.25, "max_grad_norm": 1.0,
                  "fp16": False, "eval_every": 2, "log_every": 1, "num_workers": 0, "selection_metric": "accuracy"},
        "latency": {"warmup": 1, "n_batch1": 3, "n_batches": 2}, "model": {"arch": "B", "hyper_layers": 1}}


def run(out):
    data = {"train": tiny_records(10), "validation": tiny_records(6, 100), "dev_id": tiny_records(4, 200)}
    model = Scorer(tiny_encoder(), "B", hyper_layers=1)
    return train(TCFG, data, WordTokenizer(), model, out, torch.device("cpu"), 4, 13)


def losses(out):
    return {j["step"]: j["loss"] for j in map(json.loads, (out / "train.log").read_text().splitlines()) if "loss" in j}


def test_resume_gives_the_same_training_as_an_uninterrupted_run(tmp_path, monkeypatch):
    full = run(tmp_path / "full")
    assert full["steps"] == 4 and (tmp_path / "full" / "val_probs.npz").exists()
    assert not (tmp_path / "full" / "last.pt").exists() and full["params_extra"] > 0
    import crux.train as T
    real, calls = T.evaluate_split, []

    def crash_at_second_eval(*a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real(*a, **k)

    monkeypatch.setattr(T, "evaluate_split", crash_at_second_eval)
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path / "cut")
    assert (tmp_path / "cut" / "last.pt").exists()  # saved at step 2
    monkeypatch.setattr(T, "evaluate_split", real)
    resumed = run(tmp_path / "cut")
    assert losses(tmp_path / "cut") == pytest.approx(losses(tmp_path / "full"), abs=1e-6)
    assert resumed["validation"]["accuracy"] == full["validation"]["accuracy"] and resumed["best_step"] == full["best_step"]
    p, q = (np.load(tmp_path / d / "val_probs.npz")["probs"] for d in ("full", "cut"))
    assert np.allclose(p, q, atol=1e-6)


def test_bootstrap_ci_excludes_zero_only_for_a_real_difference():
    rng = np.random.default_rng(0)
    base = (rng.random((3, 2000)) < 0.5).astype(int)
    better = np.where(rng.random((3, 2000)) < 0.1, 1, base)  # flips some misses to hits
    d, lo, hi = bootstrap_ci(better, base, 1000, 20261014, 0.95)
    assert lo > 0 and lo < d < hi
    d, lo, hi = bootstrap_ci(base, base, 1000, 20261014, 0.95)
    assert d == lo == hi == 0


def test_build_model_rejects_hyper_layers_for_a():
    with pytest.raises(ValueError):
        build_model({"arch": "A", "hyper_layers": 2}, encoder=tiny_encoder())


def test_pair_hits_count_a_pair_only_if_orig_and_edit_are_hits():
    from crux.aggregate import pair_hits
    recs = [{"meta": {"pair_id": "p1", "pair_role": "edit"}}, {"meta": {}},
            {"meta": {"pair_id": "p1", "pair_role": "orig"}}, {"meta": {"pair_id": "p2", "pair_role": "orig"}},
            {"meta": {"pair_id": "p2", "pair_role": "edit"}}]
    hits = np.array([[1, 1, 1, 1, 0], [1, 1, 0, 1, 1]])
    assert pair_hits(recs, hits).tolist() == [[1, 0], [0, 1]]


@pytest.mark.parametrize("arch,K", [("A", 0), ("B", 0), ("B", 1)])
def test_logits_of_a_record_do_not_depend_on_padding_or_on_the_other_records(arch, K):
    m = Scorer(tiny_encoder(), arch, hyper_layers=K).eval()
    if arch == "B":
        randomize_hyper(m)
    x = batch(instr=(10, 11, 12), n_opts=(3, 2))
    alone = batch(instr=(10, 11, 12), n_opts=(3,))
    wide = {k: torch.nn.functional.pad(v, (0, 5)) for k, v in alone.items() if k != "option_mask"}  # extra [PAD] columns
    other = batch(instr=(40, 41, 42), n_opts=(2, 3))  # record 0 of this batch is different
    with torch.no_grad():
        ref = m(**alone)[0]
        assert torch.allclose(m(**x)[0], ref, atol=1e-5)
        assert torch.allclose(m(**wide, option_mask=alone["option_mask"])[0], ref, atol=1e-5)
        mixed = {k: torch.cat([other[k][:1], x[k][:1]]) if other[k].shape == x[k].shape else None for k in x}
        assert torch.allclose(m(**mixed)[1], ref, atol=1e-5)


def test_the_final_model_is_the_best_checkpoint_on_validation(tmp_path, monkeypatch):
    import crux.train as T
    real, calls = T.evaluate_split, []

    def fake(*a, **k):
        m, probs = real(*a, **k)
        calls.append(1)
        return {**m, "accuracy": {1: 0.9, 2: 0.1}.get(len(calls), m["accuracy"])}, probs  # best at the first eval

    monkeypatch.setattr(T, "evaluate_split", fake)
    model = Scorer(tiny_encoder(), "B", hyper_layers=1)
    data = {"train": tiny_records(10), "validation": tiny_records(6, 100), "dev_id": tiny_records(4, 200)}
    out = train(TCFG, data, WordTokenizer(), model, tmp_path, torch.device("cpu"), 4, 13)
    best = torch.load(tmp_path / "best.pt")
    assert out["best_step"] == best["step"] == 2
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in best["model"].items())
