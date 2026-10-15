# Unified Decision Schema (Step 2)

Every dataset becomes JSONL files of decision records, one record per line. Code: `crux/schema.py`
(schema, validation, storage), `crux/data/batching.py` (tokenization, collation, batching),
`configs/step2_data.yaml` (max lengths, token budget).

## Record

| Field | Type | Content |
|---|---|---|
| `id` | str | Unique in its file. Keep the original id of the source when it exists. |
| `source` | str | Dataset name, `[a-z0-9_]+`, for example `ruletaker`. It is also the output folder name. |
| `task_type` | str | `binary`, `multi_choice`, or `scoring` (see below). |
| `state` | str | Facts and evidence. Not empty. |
| `instruction` | str | Rules and question. Not empty. |
| `options` | list[str] | The candidate answers. 2 or more. No duplicates (the check ignores case and outer spaces). |
| `target` | int or list[float] | The option index, or one score per option (`scoring`). |
| `meta` | dict | Optional data. Step 4 uses these keys for the splits: `template_id`, `composition_id`, `vocabulary_id` (str, from Step 3), `depth` (int, the depth of the question), `dataset_depth` (int, the depth of the source dataset folder), `label` (str). If `label` exists, it must equal `options[target]` (the check ignores case and outer spaces). Other keys are free, for example `theory_id` or `source_label` (the raw label when its text differs from the option). |

`state` and `instruction` are always separate fields. Architecture D encodes and caches them independently.
Records with the same `state` text can share one cached state encoding.

## Task types

- **binary**: a decision with a fixed label set for the whole source. 2 options, or 3 options when the third
  option is the "Unknown" class of open-world data (the check requires "unknown", case-folded).
  `target` is the index of the correct option. All sources use one canonical order: the positive label first,
  then the negative label, then "Unknown". RuleTaker and ProofWriter: `["True", "False"]` or
  `["True", "False", "Unknown"]`. BoardgameQA: `["proved", "disproved"]` or `["proved", "disproved", "unknown"]`.
  DynaBench: `["compliant", "not compliant"]`. Put the label text in `meta.label`.
- **multi_choice**: the options change from record to record (Step 3 uses 2 to 8 options, Super-NaturalInstructions
  uses the task options). `target` is the index of the correct option.
- **scoring**: each option has a graded value, not one correct answer. `target` is a list of finite floats,
  one per option. A higher score means a better option. Step 2 does not fix the loss: Step 5 chooses
  it (for example, soft-target cross-entropy). The accuracy uses the option with the highest score.
  Why this form: steps.md names a scoring task type but no current dataset needs more. One score per option
  is the smallest form that the candidate scorer of D and the per-option heads of A, B, and C can read.

All task types share the candidate form: a list of options and a target that points into it.

## Input modes

Both modes tokenize each field without special tokens, then add `[CLS]` and `[SEP]`. Max lengths come from
`configs/step2_data.yaml`.

Truncation never removes the question. The code splits the instruction at the last `Question:` marker into
the rules and the question. It cuts the rules from their end and keeps the question whole. Only a question
that alone is longer than the limit loses tokens (from its end). An instruction without a marker is cut from
its end. `EncodedDataset.truncated` (also `loader.dataset.truncated`) counts the truncated records per part:
`state`, `rules`, `question` (question lost), and `option`. The code does not hide truncation.

- **joint** (A, B, C), `joint_options=False`: one sequence per record, `[CLS] state [SEP] instruction [SEP]`.
  Batch: `input_ids`, `attention_mask` `[B, L]`. Only for `binary` records (a class head over the fixed options).
  All records of the dataset must have the same `options` list. Otherwise the code raises an error,
  because the class head reads `target` as a fixed class id.
- **joint**, `joint_options=True`: one sequence per option, `[CLS] state [SEP] instruction [SEP] option [SEP]`.
  Batch: `input_ids`, `attention_mask` `[B, K, L]`, `option_mask` `[B, K]`. For all task types.
  In both joint forms the total length is at most `max_length`. If the input is too long, the code cuts the
  longer field first (state or instruction), but it keeps the question whole when it fits.
  All options of a record keep the same state and instruction tokens.
- **separate** (D): `state_ids`/`state_mask` `[B, Ls]`, `instruction_ids`/`instruction_mask` `[B, Li]`,
  `option_ids`/`option_attention_mask` `[B, K, Lo]`, `option_mask` `[B, K]`. Each field is its own
  `[CLS] text [SEP]` sequence with its own max length.

`K` is the largest number of options in the batch. A padded option has only pad ids and `option_mask` False.
Encode only `option_ids[option_mask]`: a padded option has an attention mask with only zeros.
Every batch also has `labels` `[B]` (option index; -100 for `scoring`), `scores` and `score_mask` `[B, K]`
(the scoring targets), and `ids`. `max_option_length` applies to the options in both modes.

## Batching

`make_dataloader(records, tokenizer, cfg, mode, joint_options=False, train=False, seed=0)`.

- **Training** (`train=True`): `FixedBatchSampler` gives batches of `train_batch_size` records (the last batch
  can be smaller). It shuffles all records with the seed and the epoch. The batches do not depend on the
  mode, so all modes and architectures get the same records in the same steps, and `len(loader)` is
  `ceil(n / train_batch_size)` in all modes. This satisfies the equal-budget rule of steps.md (same number of
  steps, same data). Call `loader.batch_sampler.set_epoch(epoch)` for each epoch.
- **Evaluation** (`train=False`): `TokenBudgetSampler` sorts the records by padded size. It adds records to a
  batch while the padded batch size is at most `max_tokens` and the batch has at most `max_batch_size` records.
  The padded batch size uses the largest value of each dimension in the batch: `B x L` (joint),
  `B x K x L` (joint_options), or `B x (Ls + Li + K x Lo)` (separate). The order is fixed.
  Do not use it for training: the number of records per step depends on the mode.

Both samplers are for one process only. They do not split the batches between DDP ranks.

Show the batches per mode: `python -m crux.data.batching data/processed/ruletaker/depth3/dev.jsonl`.

Truncated records with the limits of `configs/step2_data.yaml` (`max_length` 640, `max_state_length` 256,
`max_instruction_length` 448, `max_option_length` 32), as state / rules / question lost (lower is better).
No option is truncated. Joint mode without options does not apply to Super-NaturalInstructions (multi_choice).

| Source (records) | joint | joint_options | separate |
|---|---|---|---|
| RuleTaker depth 3, train + dev (80,052) | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| ProofWriter OWA depth 3, train + dev (79,690) | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| BoardgameQA, 45 files (37,500) | 0 / 752 / 0 | 0 / 772 / 0 | 45 / 1,807 / 0 |
| DynaBench (543) | 136 / 314 / 0 | 136 / 314 / 0 | 194 / 301 / 0 |
| Super-NaturalInstructions (142,350) | n/a | 397 / 0 / 0 | 12,533 / 0 / 0 |

## Converter contract

One converter is one function in `crux/data/<source>.py`, with a `main()` that writes the files:

```python
def to_records(data_dir: Path, split: str, ...source options) -> list[dict]:
    """Read the raw files of one split. Return schema records."""

# in main():
from crux.schema import PROCESSED_DIR, write_jsonl
write_jsonl(records, PROCESSED_DIR / "<source>" / "<setting>" / f"{split}.jsonl")
```

- Outputs go to `data/processed/<source>/<setting>/<split>.jsonl` (git-ignored). The setting folder keeps
  different settings apart, so one run does not overwrite another: `ruletaker/depth3/`,
  `proofwriter/OWA-depth3/`, `boardgameqa/<subset>/`. A source with one setting has no setting folder
  (`dynabench/`, `supernat/`).
- RuleTaker and ProofWriter use depth 3 by default (questions of depth 0 to 3, the standard RuleTaker
  "train on depth 3 or less" setting). Step 4 keeps ProofWriter depth 5, Birds-Electricity, and ParaRules
  for the test only. Do not make train or dev data from them. Use `--depth` (and `--variant` for
  ProofWriter) for other settings.
- `write_jsonl` validates every record and the unique ids. It writes nothing if one record is bad.
- Check a file and print the counts per source, task type, number of options, and label:
  `python -m crux.schema data/processed/<source>/<setting>/<split>.jsonl`.
- Add one test with an inline raw sample (no network): check the `state`/`instruction` split, the options,
  the target, and call `validate`.
- Do not open a test split before the protocol permits it. The RuleTaker and ProofWriter converters refuse `test`.
- Reference converter: `to_records(data_dir, split, depth=3)` in `crux/data/ruletaker.py`.
  Run it with `python -m crux.data.ruletaker`.

## Examples

binary (RuleTaker depth 3, dev; the facts come from `meta-dev.jsonl` triples, the rules from its rules,
both in the order of the original context):

```json
{"id": "RelNeg-D3-1238-1", "source": "ruletaker", "task_type": "binary",
 "state": "The squirrel sees the cow. The cow visits the squirrel. The dog is not rough. The rabbit is round.",
 "instruction": "If something chases the cow and it is not green then the cow is rough. ... Question: The rabbit is round.",
 "options": ["True", "False"], "target": 0,
 "meta": {"depth": 0, "dataset_depth": 3, "label": "true", "theory_id": "RelNeg-D3-1238", "strategy": "proof", "split": "dev"}}
```

multi_choice (made-up example in the form of Step 3):

```json
{"id": "syn-000123", "source": "synthetic", "task_type": "multi_choice",
 "state": "Bob has 3 coins. Carl has 1 coin. Dana has 5 coins.",
 "instruction": "A person can buy the ticket if the person has 2 coins or more, except Dana. Who can buy the ticket?",
 "options": ["Bob", "Carl", "Dana"], "target": 0,
 "meta": {"template_id": "threshold_exception_03", "composition_id": "c17", "vocabulary_id": "v2"}}
```

scoring (made-up example):

```json
{"id": "syn-000456", "source": "synthetic", "task_type": "scoring",
 "state": "Route A takes 20 minutes. Route B takes 35 minutes. Route C is closed.",
 "instruction": "Score each route: 1 for the fastest open route, 0 for a closed route, 0.5 otherwise.",
 "options": ["Route A", "Route B", "Route C"], "target": [1.0, 0.5, 0.0],
 "meta": {"template_id": "route_score_01"}}
```
