---
license: other
license_name: mixed
license_link: DATA_LICENSES.md
language:
- en
pretty_name: CRUX decision data
tags:
- rule-reasoning
- counterfactual
configs:
- config_name: crux
  data_files:
  - split: train
    path: data/train.jsonl
  - split: dev_id
    path: data/dev_id.jsonl
  - split: validation
    path: data/validation.jsonl
  - split: test
    path: data/test.jsonl
- config_name: boardgameqa
  data_files:
  - split: test
    path: data/benchmarks/boardgameqa.jsonl
- config_name: dynabench
  data_files:
  - split: test
    path: data/benchmarks/dynabench.jsonl
---

# CRUX decision data

This is the data of the CRUX research project (https://github.com/0xAnto/crux).
The models are at https://huggingface.co/0xanto/crux.
Each record is one decision: a `state` (facts), an `instruction` (rules and a question), and `options` (the candidate answers). `target` is the index of the correct option.
`docs/schema.md` in the code repository defines the record.

## Files

| File | Records | Content | License |
|---|---|---|---|
| `data/train.jsonl` | 75,646 | Synthetic counterfactual data (Step 3 generator) | Apache-2.0 |
| `data/dev_id.jsonl` | 4,162 | Synthetic, in-distribution development split | Apache-2.0 |
| `data/validation.jsonl` | 10,106 | Synthetic, unseen rule compositions (model selection) | Apache-2.0 |
| `data/test.jsonl` | 42,221 | Synthetic 10,086 (unseen rule compositions), ProofWriter open-world depth5 18,857, birds_electricity 5,270, pararules 8,008 | Synthetic: Apache-2.0. ProofWriter records: CC BY 4.0 |
| `data/benchmarks/boardgameqa.jsonl` | 15,000 | BoardgameQA, converted | CC BY |
| `data/benchmarks/dynabench.jsonl` | 543 | DynaBench test set of the DynaGuard paper, converted | MIT |
| `data/benchmarks/supernat_ids.jsonl` | 4,100 | Super-NaturalInstructions: record IDs, task IDs, and SHA-256 hashes only. No instance text | The license of each task (`supernat_tasks.json`) |

The project read the test split one time, in Step 8. Do not use it for model selection.
`splits_manifest.json` gives the SHA-256 of each source split file. `MANIFEST.json` gives the SHA-256 of every file in this repository.
To get the Super-NaturalInstructions records, download them from the official source and match them with the hashes.

## Attribution

`NOTICE` gives the attribution of ProofWriter, RuleTaker, BoardgameQA, DynaBench, and Super-NaturalInstructions.
`DATA_LICENSES.md` records the license check of each source.

## Limitations

- The ProofWriter sets are the open-world versions (True / False / Unknown). Compare results only with published open-world results.
- The sentence frames and names of the ProofWriter depth5 set occur in the Ettin pretraining data.
- The synthetic data comes from one generator, with fixed templates and vocabularies.
