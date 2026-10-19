"""Step 2: tokenization, collation, and batching of schema records (see docs/schema.md).

Modes:
- "joint" (A, B, C): [CLS] state [SEP] instruction [SEP], and with joint_options also option [SEP].
- "separate" (D): state, instruction, and each option are separate sequences [CLS] text [SEP].
Training uses batches of a fixed number of records. Evaluation uses a token budget.
Show the batches per mode: python -m crux.data.batching data/processed/ruletaker/depth3/dev.jsonl
"""
import argparse
import random
from collections import Counter
from functools import partial

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

MODES = ("joint", "separate")
QUESTION_MARKER = "Question:"  # every converter ends the instruction with "Question: ..."


def split_question(text):
    """Return (rules, question). The question starts at the last QUESTION_MARKER. Without a marker it is ""."""
    k = text.rfind(QUESTION_MARKER)
    if k < 0:
        return text, ""
    rules = text[:k].rstrip()
    return rules, text[len(rules):]  # the space before the marker stays with the question


def _cut(ids, n, field, cut):
    """Keep the first n tokens. Record the field in `cut` if tokens are removed."""
    if len(ids) > n:
        cut.add(field)
    return ids[:max(n, 0)]


def _cut_instruction(rules, question, n, cut):
    """Keep at most n instruction tokens. Cut the rules from their end. Keep the question whole.

    Only a question that alone is longer than n loses tokens (from its end). `cut` records it as "question".
    """
    n = max(n, 0)
    if len(question) > n:
        _cut(rules, 0, "rules", cut)
        return _cut(question, n, "question", cut)
    return _cut(rules, n - len(question), "rules", cut) + question


def _row_size(mode, joint_options):
    """Return the function that gives the padded tokens of one batch row from the max shape of the batch."""
    if mode == "separate":
        return lambda s: s[0] + s[1] + s[2] * s[3]  # Ls + Li + K x Lo
    return (lambda s: s[0] * s[1]) if joint_options else (lambda s: s[0])  # K x L, or L


class EncodedDataset(Dataset):
    """Token ids of schema records for one mode.

    `truncated` counts the records with removed tokens: "state", "rules" (the instruction before the
    question), "question" (the question alone is over the limit), and "option".
    `shapes` holds the padded dimensions of each item: (L,) for joint, (K, L) for joint_options,
    (Ls, Li, K, Lo) for separate. `row_size(shape)` gives the tokens of one padded batch row.
    """

    def __init__(self, records, tokenizer, cfg, mode, joint_options=False):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
        if mode == "joint" and not joint_options:
            kinds = {tuple(r["options"]) for r in records}
            if len(kinds) > 1:
                raise ValueError(f"joint mode without options needs one options tuple for all records "
                                 f"(the class head reads the target as a fixed class id). Found {len(kinds)}: "
                                 f"{sorted(kinds)[:3]}. Use joint_options=True or one file per options tuple.")
        self.mode, self.joint_options = mode, joint_options
        self.row_size = _row_size(mode, joint_options)
        self.pad_id = tokenizer.pad_token_id
        cls, sep = [tokenizer.cls_token_id], [tokenizer.sep_token_id]
        # Tokenize each distinct text one time. Many records share one state.
        parts = {r["instruction"]: split_question(r["instruction"]) for r in records}
        texts = sorted({t for r in records for t in (r["state"], *parts[r["instruction"]], *r["options"])})
        tok = dict(zip(texts, tokenizer(texts, add_special_tokens=False)["input_ids"]))
        self.items, self.shapes = [], []
        self.truncated = Counter(records=0, state=0, rules=0, question=0, option=0)
        for r in records:
            if mode == "joint" and not joint_options and r["task_type"] != "binary":
                raise ValueError(f"record {r['id']}: joint mode without options supports binary records only")
            cut = set()
            opts = [_cut(tok[o], cfg["max_option_length"] - 2, "option", cut) for o in r["options"]]
            state = tok[r["state"]]
            rules, question = (tok[t] for t in parts[r["instruction"]])
            if mode == "joint":
                # Room for the state and the instruction. Use the longest option, so all options share them.
                room = max(cfg["max_length"] - 3 - (max(len(o) + 1 for o in opts) if joint_options else 0), 0)
                # Longest first: cut the longer field, but keep the question whole when it fits in the room.
                n_instr = len(rules) + len(question)
                keep_instr = min(n_instr, max(room - len(state), (room + 1) // 2, min(len(question), room)))
                state = _cut(state, room - keep_instr, "state", cut)
                head = cls + state + sep + _cut_instruction(rules, question, keep_instr, cut) + sep
                if joint_options:
                    item = {"input_ids": [head + o + sep for o in opts]}
                    shape = (len(opts), max(map(len, item["input_ids"])))
                else:
                    item, shape = {"input_ids": head}, (len(head),)
            else:
                item = {"state_ids": cls + _cut(state, cfg["max_state_length"] - 2, "state", cut) + sep,
                        "instruction_ids": cls + _cut_instruction(rules, question, cfg["max_instruction_length"] - 2,
                                                                  cut) + sep,
                        "option_ids": [cls + o + sep for o in opts]}
                shape = (len(item["state_ids"]), len(item["instruction_ids"]), len(opts),
                         max(map(len, item["option_ids"])))
            item.update(id=r["id"], task_type=r["task_type"], target=r["target"])
            self.items.append(item)
            self.shapes.append(shape)
            self.truncated.update(cut)
            self.truncated["records"] += bool(cut)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def _pad(seqs, pad_id):
    """Pad a list of id lists to one LongTensor. Return (ids, attention_mask)."""
    width = max(map(len, seqs))
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        mask[i, :len(s)] = 1
    return ids, mask


def _pad_options(groups, pad_id):
    """Pad a list of option lists to [B, K, L]. Padded options have only pad ids and option_mask False."""
    k = max(map(len, groups))
    flat_ids, flat_mask = _pad([s for g in groups for s in g], pad_id)
    ids = torch.full((len(groups), k, flat_ids.shape[1]), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    option_mask = torch.zeros((len(groups), k), dtype=torch.bool)
    row = 0
    for b, g in enumerate(groups):
        ids[b, :len(g)] = flat_ids[row:row + len(g)]
        mask[b, :len(g)] = flat_mask[row:row + len(g)]
        option_mask[b, :len(g)] = True
        row += len(g)
    return ids, mask, option_mask


def collate(items, mode, pad_id, joint_options=False):
    """Build one batch.

    Targets: `labels` [B] is the option index, or -100 for a scoring record (cross_entropy ignores -100).
    `scores` [B, K] holds the scoring targets. `score_mask` [B, K] is True where a score exists.
    """
    if mode == "joint" and not joint_options:
        batch = dict(zip(("input_ids", "attention_mask"), _pad([it["input_ids"] for it in items], pad_id)))
        k = 1
    elif mode == "joint":
        ids, mask, option_mask = _pad_options([it["input_ids"] for it in items], pad_id)
        batch = {"input_ids": ids, "attention_mask": mask, "option_mask": option_mask}
        k = ids.shape[1]
    else:
        s_ids, s_mask = _pad([it["state_ids"] for it in items], pad_id)
        i_ids, i_mask = _pad([it["instruction_ids"] for it in items], pad_id)
        o_ids, o_mask, option_mask = _pad_options([it["option_ids"] for it in items], pad_id)
        batch = {"state_ids": s_ids, "state_mask": s_mask, "instruction_ids": i_ids, "instruction_mask": i_mask,
                 "option_ids": o_ids, "option_attention_mask": o_mask, "option_mask": option_mask}
        k = o_ids.shape[1]
    batch["labels"] = torch.tensor([-100 if it["task_type"] == "scoring" else it["target"] for it in items])
    batch["scores"] = torch.zeros((len(items), k))
    batch["score_mask"] = torch.zeros((len(items), k), dtype=torch.bool)
    for b, it in enumerate(items):
        if it["task_type"] == "scoring":
            batch["scores"][b, :len(it["target"])] = torch.tensor(it["target"], dtype=torch.float)
            batch["score_mask"][b, :len(it["target"])] = True
    batch["ids"] = [it["id"] for it in items]
    return batch


class TokenBudgetSampler(Sampler):
    """Evaluation batch sampler with a budget on the padded batch size. The order is fixed.

    It sorts the items by padded row size. A batch grows while records x row_size(max shape) <= max_tokens
    and records <= max_batch_size. The max shape holds the largest value of each dimension in the batch,
    so the budget applies to the real padded tensors (B x L, B x K x L, or B x (Ls + Li + K x Lo)).
    An item that alone is over the budget gets its own batch.
    For one process only: it does not split the batches between DDP ranks.
    """

    def __init__(self, shapes, row_size, max_tokens, max_batch_size=None):
        order = sorted(range(len(shapes)), key=lambda i: (row_size(shapes[i]), shapes[i]))
        self.batches, batch, top = [], [], None
        for i in order:
            new_top = tuple(map(max, top, shapes[i])) if batch else shapes[i]
            full = max_batch_size and len(batch) >= max_batch_size
            if batch and (full or (len(batch) + 1) * row_size(new_top) > max_tokens):
                self.batches.append(batch)
                batch, new_top = [], shapes[i]
            batch.append(i)
            top = new_top
        if batch:
            self.batches.append(batch)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class FixedBatchSampler(Sampler):
    """Training batch sampler: batch_size records per batch, in a new random order for each epoch.

    The order depends only on the seed, the epoch, and the number of records, not on the input mode.
    Thus all modes get the same records in the same steps, and len() = ceil(n / batch_size) in all modes
    (the last batch can be smaller). Call set_epoch(epoch) at the start of each epoch.
    For one process only: it does not split the batches between DDP ranks.
    """

    def __init__(self, n, batch_size, seed=0):
        self.n, self.batch_size, self.seed, self.epoch = n, batch_size, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        idx = list(range(self.n))
        random.Random(self.seed * 1_000_003 + self.epoch).shuffle(idx)
        return iter([idx[i:i + self.batch_size] for i in range(0, self.n, self.batch_size)])

    def __len__(self):
        return -(-self.n // self.batch_size)


class UnitBatchSampler(FixedBatchSampler):
    """Paired counterfactual training (Step 6): FixedBatchSampler, but it shuffles whole units, not records.

    `units` holds the record indices of each unit, in pair order (orig, edit, orig, edit, ...). The sampler
    puts the units in a new random order for each epoch, joins them, and cuts batches of batch_size records.
    A unit has an even number of records, so with an even batch_size a cut never separates the two members
    of a pair. A cut can separate the paraphrases of a unit. The records and len() are those of FixedBatchSampler.
    """

    def __init__(self, units, batch_size, seed=0):
        super().__init__(sum(map(len, units)), batch_size, seed)
        self.units = units

    def __iter__(self):
        units = list(self.units)
        random.Random(self.seed * 1_000_003 + self.epoch).shuffle(units)
        idx = [i for u in units for i in u]
        return iter([idx[i:i + self.batch_size] for i in range(0, self.n, self.batch_size)])


def pair_units(records):
    """Record indices of each unit (`meta.unit_id`), in pair order: the orig and the edit record of each pair
    side by side. A record without a unit_id is a unit of its own."""
    units = {}
    for i, r in enumerate(records):
        m = r.get("meta", {})
        units.setdefault(m.get("unit_id", r["id"]), []).append((m.get("pair_id", ""), m.get("pair_role") != "orig", i))
    return [[i for *_, i in sorted(u)] for u in units.values()]


def make_dataloader(records, tokenizer, cfg, mode, joint_options=False, train=False, seed=0, paired=False):
    """Return a DataLoader of padded batches. cfg holds the keys of configs/step2_data.yaml.

    train=True: FixedBatchSampler with cfg["train_batch_size"] records per batch (equal budget for all
    modes), or with paired=True the UnitBatchSampler (Step 6). Call loader.batch_sampler.set_epoch(epoch)
    at the start of each epoch.
    train=False: TokenBudgetSampler with cfg["max_tokens"] (evaluation).
    loader.dataset.truncated holds the truncation counts. One process only (no DDP).
    """
    ds = EncodedDataset(records, tokenizer, cfg, mode, joint_options)
    if train and paired:
        sampler = UnitBatchSampler(pair_units(records), cfg["train_batch_size"], seed)
    elif train:
        sampler = FixedBatchSampler(len(ds), cfg["train_batch_size"], seed)
    else:
        sampler = TokenBudgetSampler(ds.shapes, ds.row_size, cfg["max_tokens"], cfg.get("max_batch_size"))
    return DataLoader(ds, batch_sampler=sampler,
                      collate_fn=partial(collate, mode=mode, pad_id=ds.pad_id, joint_options=joint_options))


def main():
    import yaml
    from transformers import AutoTokenizer

    from crux.schema import ROOT, read_jsonl

    ap = argparse.ArgumentParser(description="Build the batches of each mode from a schema JSONL file.")
    ap.add_argument("file")
    ap.add_argument("--config", default=str(ROOT / "configs" / "step2_data.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    records = read_jsonl(args.file)
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    for mode, joint_options in (("joint", False), ("joint", True), ("separate", False)):
        try:
            loader = make_dataloader(records, tokenizer, cfg, mode, joint_options)
        except ValueError as e:
            print(f"{mode} joint_options={joint_options}: not possible: {e}")
            continue
        train = make_dataloader(records, tokenizer, cfg, mode, joint_options, train=True, seed=13)
        shapes = {k: tuple(v.shape) for k, v in next(iter(train)).items() if torch.is_tensor(v)}
        print(f"{mode} joint_options={joint_options}: {len(train)} train batches, {len(loader)} eval batches; "
              f"truncated {dict(loader.dataset.truncated)} of {len(records)}; first train batch {shapes}")


if __name__ == "__main__":
    main()
