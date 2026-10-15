import torch

import pytest

from crux.data.batching import EncodedDataset, make_dataloader

CLS, SEP, PAD = 1, 2, 0
CFG = {"max_length": 12, "max_state_length": 6, "max_instruction_length": 6, "max_option_length": 4,
       "max_tokens": 1000, "max_batch_size": 8, "train_batch_size": 4}


class WordTokenizer:
    """One id per distinct word. No network."""
    cls_token_id, sep_token_id, pad_token_id = CLS, SEP, PAD

    def __init__(self):
        self.vocab = {}

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [[self.vocab.setdefault(w, len(self.vocab) + 10) for w in t.split()] for t in texts]}


def rec(i, options, target=0, task_type="multi_choice", state="s1 s2", instruction="q1 q2"):
    return {"id": f"r{i}", "source": "t", "task_type": task_type, "state": state, "instruction": instruction,
            "options": options, "target": target, "meta": {}}


def test_separate_mode_pads_options_and_masks_them():
    records = [rec(0, ["a", "b"], 1), rec(1, ["c", "d e f", "g", "h"], 2),
               rec(2, ["x", "y", "z"], [0.1, 0.9, 0.5], task_type="scoring")]
    loader = make_dataloader(records, WordTokenizer(), CFG, "separate")
    batch = next(iter(loader))
    order = [int(i[1]) for i in batch["ids"]]  # the sampler sorts by length
    n_opts = torch.tensor([len(records[i]["options"]) for i in order])
    # 4 options at most. max_option_length 4 cuts "d e f" to [CLS] d e [SEP].
    assert batch["option_ids"].shape == (3, 4, 4) and loader.dataset.truncated["option"] == 1
    assert batch["option_mask"].tolist() == [[k < n for k in range(4)] for n in n_opts]
    pad_rows = ~batch["option_mask"]
    assert (batch["option_ids"][pad_rows] == PAD).all() and (batch["option_attention_mask"][pad_rows] == 0).all()
    assert (batch["option_ids"][batch["option_mask"]][:, 0] == CLS).all()
    s = order.index(2)
    assert batch["labels"][s] == -100 and batch["score_mask"][s].tolist() == [True, True, True, False]
    assert batch["scores"][s, :3].tolist() == torch.tensor([0.1, 0.9, 0.5]).tolist()
    assert batch["labels"][order.index(1)] == 2 and not batch["score_mask"][order.index(1)].any()


def decode(tok, ids):
    words = {v: k for k, v in tok.vocab.items()} | {CLS: "[CLS]", SEP: "[SEP]"}
    return [words[i] for i in ids]


def test_joint_mode_cuts_the_longer_field_first_and_counts_it():
    tok = WordTokenizer()
    long_state = " ".join(f"f{i}" for i in range(20))
    ds = EncodedDataset([rec(0, ["True", "False"], task_type="binary", state=long_state)], tok, CFG, "joint")
    # max_length 12 = [CLS] + 7 state tokens + [SEP] + the whole instruction (2) + [SEP]
    assert decode(tok, ds[0]["input_ids"]) == ["[CLS]", *(f"f{i}" for i in range(7)), "[SEP]", "q1", "q2", "[SEP]"]
    assert ds.truncated == {"records": 1, "state": 1, "rules": 0, "question": 0, "option": 0}


@pytest.mark.parametrize("mode", ["joint", "separate"])
def test_truncation_cuts_the_rules_and_keeps_the_question(mode):
    tok = WordTokenizer()
    rules = " ".join(f"r{i}" for i in range(20))
    r = rec(0, ["True", "False"], task_type="binary", state=" ".join(f"f{i}" for i in range(20)),
            instruction=f"{rules} Question: is x?")
    ds = EncodedDataset([r], tok, CFG, mode)
    words = decode(tok, ds[0]["input_ids" if mode == "joint" else "instruction_ids"])
    assert words[-4:] == ["Question:", "is", "x?", "[SEP]"] and "r0" in words and "r19" not in words
    assert ds.truncated == {"records": 1, "state": 1, "rules": 1, "question": 0, "option": 0}


def test_a_question_over_the_limit_is_counted_as_lost():
    long_question = "Question: " + " ".join(f"w{i}" for i in range(10))
    ds = EncodedDataset([rec(0, ["a", "b"], instruction=f"r1 {long_question}")], WordTokenizer(), CFG, "separate")
    assert len(ds[0]["instruction_ids"]) == CFG["max_instruction_length"]
    assert ds.truncated["question"] == 1 and ds.truncated["rules"] == 1


def test_joint_options_repeats_the_head_for_each_option_and_masks_padding():
    tok = WordTokenizer()
    records = [rec(0, ["a", "b"], 1), rec(1, ["c", "d", "e"], 2)]
    batch = next(iter(make_dataloader(records, tok, CFG, "joint", joint_options=True)))
    assert batch["input_ids"].shape == (2, 3, 9)  # [CLS] s1 s2 [SEP] q1 q2 [SEP] option [SEP]
    assert batch["option_mask"].tolist() == [[True, True, False], [True, True, True]]
    assert decode(tok, batch["input_ids"][1, 2].tolist()) == ["[CLS]", "s1", "s2", "[SEP]", "q1", "q2", "[SEP]", "e", "[SEP]"]
    assert (batch["attention_mask"][0, 2] == 0).all() and batch["labels"].tolist() == [1, 2]


def test_plain_joint_mode_refuses_mixed_option_tuples():
    records = [rec(0, ["True", "False"], task_type="binary"), rec(1, ["True", "False", "Unknown"], task_type="binary")]
    with pytest.raises(ValueError, match="one options tuple"):
        EncodedDataset(records, WordTokenizer(), CFG, "joint")


def test_separate_mode_mixes_binary_records_with_two_and_three_options():
    records = [rec(0, ["True", "False"], 1, "binary"), rec(1, ["True", "False", "Unknown"], 2, "binary")]
    batch = next(iter(make_dataloader(records, WordTokenizer(), CFG, "separate")))
    by_id = dict(zip(batch["ids"], range(2)))
    assert batch["option_mask"][by_id["r0"]].tolist() == [True, True, False]
    assert batch["option_mask"][by_id["r1"]].tolist() == [True, True, True]
    assert batch["labels"][by_id["r0"]] == 1 and batch["labels"][by_id["r1"]] == 2


@pytest.mark.parametrize("mode, joint_options", [("joint", False), ("joint", True), ("separate", False)])
def test_eval_batches_respect_the_padded_budget(mode, joint_options):
    # Short and long fields in different records: the padded tensor is larger than the sum of the lengths.
    records = [rec(i, [f"o{j}" for j in range(2 + i % 3)], 0, "binary" if mode == "joint" and not joint_options
                   else "multi_choice", state=" ".join(["s"] * (1 + i % 5)), instruction=" ".join(["q"] * (1 + i % 4)))
               for i in range(60)]
    if mode == "joint" and not joint_options:
        for r in records:
            r["options"] = ["True", "False"]
    cfg = {**CFG, "max_tokens": 120}
    loader = make_dataloader(records, WordTokenizer(), cfg, mode, joint_options)
    keys = ["state_ids", "instruction_ids", "option_ids"] if mode == "separate" else ["input_ids"]
    seen = []
    for batch in loader:
        seen += batch["ids"]
        assert len(batch["ids"]) == 1 or sum(batch[k].numel() for k in keys) <= cfg["max_tokens"]
    assert sorted(seen) == sorted(r["id"] for r in records) and len(loader) > 1


def test_train_batches_have_a_fixed_size_and_the_same_records_in_all_modes():
    records = [rec(i, ["True", "False"], 0, "binary", state=" ".join(["s"] * (1 + i % 7))) for i in range(10)]
    tok = WordTokenizer()
    loaders = [make_dataloader(records, tok, CFG, mode, jo, train=True, seed=13)
               for mode, jo in (("joint", False), ("joint", True), ("separate", False))]
    batches = [[b["ids"] for b in loader] for loader in loaders]
    assert [len(loader) for loader in loaders] == [3, 3, 3]
    assert [len(b) for b in batches[0]] == [4, 4, 2]
    assert batches[0] == batches[1] == batches[2]
    assert batches[0] == [b["ids"] for b in make_dataloader(records, tok, CFG, "joint", train=True, seed=13)]
    loaders[0].batch_sampler.set_epoch(1)
    assert [b["ids"] for b in loaders[0]] != batches[0]
