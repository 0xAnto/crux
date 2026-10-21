import copy
import json
import math

import numpy as np
import pytest
import torch
from transformers import ModernBertConfig, ModernBertModel

from crux.aggregate import bootstrap_ci
from crux.models import Hypernet, Lora, Scorer, build_model, count_params, span_mean, static_ranks
from crux.data.batching import EncodedDataset, collate
from crux.train import consistency_loss, input_mode, load_config, paraphrase_rows, train

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
                     "max_batch_size": 8, "max_state_length": 12, "max_instruction_length": 12},
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


def reference_models():
    """crux/models.py of commit b7220bf (before the bound on the encoder adapters), loaded as a module."""
    import subprocess
    import types
    src = subprocess.run(["git", "show", "b7220bf:crux/models.py"], capture_output=True, text=True, check=True).stdout
    mod = types.ModuleType("models_b7220bf")
    exec(compile(src, "models_b7220bf", "exec"), mod.__dict__)
    return mod


@pytest.mark.parametrize("arch,K", [("A", 0), ("B", 0), ("A-param", 0), ("A-param", 2)])
def test_outputs_and_gradients_without_encoder_adapters_equal_commit_b7220bf(arch, K):
    ref_cls = reference_models().Scorer
    ref = ref_cls(tiny_encoder(), arch, hyper_layers=K)
    new = Scorer(tiny_encoder(), arch, hyper_layers=K)
    if arch == "B":
        randomize_hyper(ref)
    new.load_state_dict(ref.state_dict())
    x, grads = batch(), []
    for m in (ref, new):
        m.train(), m.zero_grad()
        m(**x)[x["option_mask"]].square().sum().backward()
        grads.append({k: p.grad.clone() for k, p in m.named_parameters() if p.grad is not None})
    assert torch.equal(ref(**x), new(**x))
    assert grads[0].keys() == grads[1].keys() and all(torch.equal(grads[0][k], grads[1][k]) for k in grads[0])


def test_encoder_adapter_factors_are_bounded():
    m = Scorer(tiny_encoder(), "B", hyper_layers=2).eval()
    with torch.no_grad():
        m.hyper.fc2.bias.add_(100.0)  # huge generated factors
        m.hyper.fc2.weight.normal_(std=10.0)
    seen = []
    for lora in m.adapted:
        lora.register_forward_hook(lambda mod, inp, out: seen.append(mod.delta))
    m(**batch())
    assert len(seen) == len(m.adapted) == 4
    for lora, (a, b) in zip(m.adapted, seen):
        assert a.abs().max() <= lora.in_features ** -0.5 + 1e-6 and b.abs().max() <= lora.out_features ** -0.5 + 1e-6


def test_consistency_loss_is_the_mean_js_divergence_of_a_group_and_ignores_padded_options():
    p = torch.tensor([[0.8, 0.2], [0.2, 0.8]])
    expected = 0.8 * math.log(1.6) + 0.2 * math.log(0.4)  # KL of each row to the mean [0.5, 0.5]; equal for both rows
    logits = torch.cat([p.log(), torch.full((2, 1), -torch.inf)], 1).requires_grad_()
    mask = torch.tensor([[True, True, False]] * 2)
    loss = consistency_loss(logits, mask, [[0, 1]])
    loss.backward()
    assert loss.item() == pytest.approx(expected, abs=1e-6)
    assert torch.isfinite(logits.grad[:, :2]).all() and logits.grad[:, :2].abs().sum() > 0
    assert consistency_loss(p.log(), mask[:, :2], [[0, 1]]).item() == pytest.approx(expected, abs=1e-6)
    assert consistency_loss(p.log()[[0, 0]], mask[:, :2], [[0, 1]]).item() == pytest.approx(0, abs=1e-7)
    assert consistency_loss(p.log(), mask[:, :2], []).item() == 0


def test_paraphrase_rows_returns_only_the_groups_with_two_rows_in_the_batch():
    group_of = {"a": "g1", "b": "g1", "c": "g2", "d": None, "e": "g3", "f": "g3", "g": "g3"}
    assert paraphrase_rows(["a", "c", "b", "d", "e", "x"], group_of) == [[0, 2]]
    assert paraphrase_rows(["e", "a", "f", "g"], group_of) == [[0, 2, 3]]


def test_step6_architectures_set_the_train_keys_and_keep_the_step5_defaults_of_other_keys():
    c = load_config("configs/step6.yaml", "C", 0)
    assert c["model"]["arch"] == "B" and c["train"]["paired"] is True and c["train"]["consistency_weight"] == 1.0
    for arch, model_arch in (("A+CF", "A"), ("B+CF", "B")):
        c = load_config("configs/step6.yaml", arch, 0)
        assert c["model"]["arch"] == model_arch and c["train"]["paired"] is True
        assert c["train"]["consistency_weight"] == 0.0 and c["train"]["lr"] == 3.0e-5
    assert "train" not in c["model"]


def paired_records(n_units):
    """Units of 4 records (2 pairs): the 2 orig records and the 2 edit records are paraphrases of each other."""
    recs = []
    for u in range(n_units):
        for p in range(2):
            for role in ("orig", "edit"):
                meta = {"unit_id": f"u{u}", "pair_id": f"u{u}-t{p}", "pair_role": role, "paraphrase_group": f"u{u}-{role}"}
                recs.append({"id": f"r{len(recs)}", "source": "t", "task_type": "multi_choice",
                             "state": f"s{p} s{u % 3} s1", "instruction": f"i{u % 2} i1 Question: q{u}",
                             "options": ["o0", "o1", "o2"], "target": (u + p) % 3, "meta": meta})
    return recs


def test_train_with_paired_batches_adds_the_consistency_loss_only_for_a_positive_weight(tmp_path):
    runs = {}
    for weight in (0.0, 1.0):
        cfg = {**TCFG, "train": {**TCFG["train"], "paired": True, "consistency_weight": weight}}
        data = {"train": paired_records(3), "validation": tiny_records(6, 100), "dev_id": tiny_records(4, 200)}
        out = train(cfg, data, WordTokenizer(), Scorer(tiny_encoder(), "B", hyper_layers=1), tmp_path / str(weight),
                    torch.device("cpu"), 3, 13)
        log = [j for j in map(json.loads, (tmp_path / str(weight) / "train.log").read_text().splitlines()) if "loss" in j]
        assert out["paired"] is True and out["consistency_weight"] == weight
        assert all(math.isfinite(j["loss"]) for j in log)
        runs[weight] = (log, np.load(tmp_path / str(weight) / "val_probs.npz")["probs"])
    assert all("consistency" not in j for j in runs[0.0][0])
    assert all(math.isfinite(j["consistency"]) and j["consistency"] > 0 for j in runs[1.0][0])
    assert not np.allclose(runs[0.0][1], runs[1.0][1], atol=1e-6)  # the consistency loss changes the weights


# Step 7: the split encoding (joint_layers) and the set-attention block (option_attention).

def randomize_block(model, std=0.3):
    """Give the set-attention block non-zero weights (it starts as the identity), so that the tests are not vacuous."""
    torch.manual_seed(2)
    for p in model.option_attn.parameters():
        p.data.normal_(std=std)


SHARED_TOKENIZER = WordTokenizer()  # one vocabulary: the same word has the same id in every batch of a test


def real_batch(records, mode):
    """One batch of the records in the given mode, from the real tokenization and collation code."""
    ds = EncodedDataset(records, SHARED_TOKENIZER, TCFG["data_cfg"], mode, True)
    return collate([ds[i] for i in range(len(ds))], mode, ds.pad_id, True)  # in the order of the records


def d_model(J, attn=True, arch="B", layers=4):
    m = Scorer(tiny_encoder(layers), arch, joint_layers=J, option_attention=attn).eval()
    if arch == "B":
        randomize_hyper(m)
    if attn:
        randomize_block(m)
    return m


def test_cache_gives_the_same_logits_as_encoding_the_state_and_the_instruction_again():
    m = d_model(2)
    x = real_batch(tiny_records(8), "separate")
    with torch.no_grad():
        ref = m(**x)
        states = m.encode_bottom(x["state_ids"], x["state_mask"])
        instructions = m.encode_bottom(x["instruction_ids"], x["instruction_mask"])
        for kw in ({"state_cache": states}, {"instruction_cache": instructions},
                   {"state_cache": states, "instruction_cache": instructions}):
            assert torch.allclose(m(**{**x, **kw}), ref, atol=1e-5)
        # The cache replaces the ids: wrong ids with the cache give the same logits.
        wrong = {**x, "state_ids": x["state_ids"].flip(0), "instruction_ids": x["instruction_ids"].flip(0)}
        assert torch.allclose(m(**wrong, state_cache=states, instruction_cache=instructions), ref, atol=1e-5)
        assert not torch.allclose(m(**wrong), ref, atol=1e-4)


@pytest.mark.parametrize("J", [0, 2])
def test_permuting_the_options_permutes_the_logits(J):
    m = d_model(J)
    mode = "separate" if J else "joint"
    rec = [tiny_records(8)[i] for i in (4, 2)]  # 3 and 4 options
    perms = [[2, 0, 1], [3, 1, 0, 2]]
    shuffled = [{**r, "options": [r["options"][p] for p in perm], "target": perm.index(r["target"])}
                for r, perm in zip(rec, perms)]
    with torch.no_grad():
        a, b = m(**real_batch(rec, mode)), m(**real_batch(shuffled, mode))
        assert torch.isfinite(a[0, :3]).all() and torch.isfinite(a[1, :4]).all()
        for i, perm in enumerate(perms):
            assert torch.allclose(b[i, :len(perm)], a[i, perm], atol=1e-5)
        plain = d_model(J, attn=False)
        plain.load_state_dict({k: v for k, v in m.state_dict().items() if "option_attn" not in k})
        assert not torch.allclose(plain(**real_batch(rec, mode))[0, :3], a[0, :3], atol=1e-4)  # the block does something


@pytest.mark.parametrize("J", [0, 2])
def test_a_padded_option_does_not_change_the_scores_of_the_real_options(J):
    m, mode = d_model(J), "separate" if J else "joint"
    recs = tiny_records(8)[:3]  # 2, 3, and 4 options
    with torch.no_grad():
        both = m(**real_batch(recs[:2], mode))  # record 0 has a padded option
        alone = m(**real_batch(recs[:1], mode))
        assert both[0, 2:].eq(-torch.inf).all()
        assert torch.allclose(both[0, :2], alone[0, :2], atol=1e-5)


@pytest.mark.parametrize("J,arch", [(0, "A"), (0, "B"), (2, "A"), (2, "B")])
def test_the_set_attention_block_is_the_identity_at_the_start(J, arch):
    base = Scorer(tiny_encoder(4), arch, joint_layers=J).eval()
    new = Scorer(copy.deepcopy(base.encoder), arch, joint_layers=J, option_attention=True).eval()
    missing = new.load_state_dict(base.state_dict(), strict=False)
    assert all(k.startswith("option_attn.") for k in missing.missing_keys) and not missing.unexpected_keys
    x = real_batch(tiny_records(8), "separate" if J else "joint")
    with torch.no_grad():
        assert torch.allclose(base(**x), new(**x), atol=1e-6)
    added = count_params(new)["params_total"] - count_params(base)["params_total"]
    assert count_params(new)["params_extra"] - count_params(base)["params_extra"] == added > 0


def test_split_encoding_does_not_depend_on_the_padding_of_the_segments():
    m = d_model(2)
    recs = tiny_records(8)[:2]
    with torch.no_grad():
        ref = m(**real_batch(recs, "separate"))
        x = real_batch(recs, "separate")
        for k in ("state_ids", "state_mask", "instruction_ids", "instruction_mask"):
            x[k] = torch.nn.functional.pad(x[k], (0, 7))  # more [PAD] columns in the state and the instruction
        x["option_ids"], x["option_attention_mask"] = (torch.nn.functional.pad(x[k], (0, 3)) for k in ("option_ids", "option_attention_mask"))
        assert torch.allclose(m(**x)[x["option_mask"]], ref[x["option_mask"]], atol=1e-5)


def test_the_split_encoding_rejects_hyper_layers_and_a_wrong_number_of_joint_layers():
    for kw in ({"hyper_layers": 1}, {"joint_layers": 5}):
        with pytest.raises(ValueError):
            Scorer(tiny_encoder(4), "B", **{"joint_layers": 2, **kw})


def test_old_architectures_keep_the_parameters_and_the_random_numbers_of_commit_b7220bf():
    ref_cls, states = reference_models().Scorer, []
    for cls in (ref_cls, Scorer):
        for arch, K in (("A", 0), ("B", 0), ("A-param", 2)):
            enc = tiny_encoder()
            torch.manual_seed(5)
            m = cls(enc, arch, hyper_layers=K)
            states.append((m.state_dict(), torch.get_rng_state()))
    for (sd_r, rng_r), (sd_n, rng_n) in zip(states[:3], states[3:]):
        assert sd_r.keys() == sd_n.keys() and all(torch.equal(sd_r[k], sd_n[k]) for k in sd_r)
        assert torch.equal(rng_r, rng_n)  # the new modules are not created, so no random number is drawn


def test_the_input_mode_follows_from_the_model_config():
    assert input_mode({"arch": "B"}) == "joint" and input_mode({"joint_layers": 0, "option_attention": True}) == "joint"
    assert input_mode({"joint_layers": 4}) == "separate"
    for arch, mode, J, attn in (("D", "separate", 4, True), ("C+cache", "separate", 4, False),
                                ("C+equivariant", "joint", 0, True), ("D-CF", "separate", 4, True)):
        c = load_config("configs/step7.yaml", arch, 0)
        assert input_mode(c["model"]) == mode and c["model"].get("joint_layers", 0) == J
        assert c["model"].get("option_attention", False) is attn and "train" not in c["model"]
        assert c["train"]["paired"] is (arch != "D-CF") and c["train"]["lr"] == 3.0e-5 and c["out_root"] == "runs/step7"


def test_separate_training_with_paired_batches_writes_order_invariance_memory_and_cache_hit_latency(tmp_path):
    cfg = {**TCFG, "model": {"arch": "B", "hyper_layers": 0, "joint_layers": 2, "option_attention": True},
           "train": {**TCFG["train"], "paired": True, "consistency_weight": 1.0}}
    data = {"train": paired_records(3), "validation": tiny_records(6, 100), "dev_id": tiny_records(4, 200)}
    model = Scorer(tiny_encoder(4), "B", joint_layers=2, option_attention=True)
    out = train(cfg, data, WordTokenizer(), model, tmp_path, torch.device("cpu"), 4, 13)
    log = [j for j in map(json.loads, (tmp_path / "train.log").read_text().splitlines()) if "loss" in j]
    assert all(math.isfinite(j["loss"]) and j["consistency"] > 0 for j in log)
    assert out["joint_layers"] == 2 and out["option_attention"] is True and out["params_extra"] > 0
    oi = out["order_invariance"]
    assert set(oi) == {"seed", "n_records", "n_moved", "argmax_agreement", "mean_abs_prob_diff", "max_abs_prob_diff",
                       "shuffled"} and "accuracy" in oi["shuffled"]  # the tiny records have no ids for unseen_rule_accuracy
    assert oi["n_records"] == 6 and oi["argmax_agreement"] == 1.0 and oi["max_abs_prob_diff"] < 1e-4  # equivariant model
    mem = out["memory"]
    assert mem["model_bytes"] == {"fp32": 4 * out["params_total"], "fp16": 2 * out["params_total"]}
    assert mem["cuda_peak_extra_bytes"] == {"batch_size_1": None, "eval_batches": None}  # CPU
    assert mem["cache_bytes_fp16"]["n_states"] == 5 and mem["cache_bytes_fp16"]["state_mean"] > 0
    hit = out["latency"]["cache_hit"]
    assert set(hit) == {"state", "state_and_instruction"} and hit["state"]["batch_size_1"]["p50_ms_per_record"] > 0
    assert {"batch_size_1", "eval_batches"} <= set(out["latency"])  # the keys of crux.aggregate stay


def test_shuffle_options_is_fixed_and_keeps_the_target_on_the_same_option():
    from crux.train import shuffle_options
    recs = tiny_records(30) + [{**tiny_records(1)[0], "id": "sc", "task_type": "scoring", "target": [0.1, 0.9]}]
    a, pa = shuffle_options(recs, 7)
    b, pb = shuffle_options(recs, 7)
    assert pa == pb and a == b and any(p != sorted(p) for p in pa)
    for r, s, perm in zip(recs, a, pa):
        assert sorted(s["options"]) == sorted(r["options"]) and s["options"] == [r["options"][p] for p in perm]
        if r["task_type"] == "scoring":
            assert s["target"] == [r["target"][p] for p in perm]
        else:
            assert s["options"][s["target"]] == r["options"][r["target"]]
