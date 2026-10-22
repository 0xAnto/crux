---
license: other
license_name: mixed
license_link: https://huggingface.co/datasets/0xanto/crux-data/blob/main/DATA_LICENSES.md
language:
- en
pretty_name: CRUX decision data
task_categories:
- text-classification
size_categories:
- 100K<n<1M
tags:
- rule-reasoning
- counterfactual
configs:
- config_name: crux
  default: true
  data_files:
  - split: train
    path: data/train.jsonl
  - split: dev_id
    path: data/dev_id.jsonl
  - split: validation
    path: data/validation.jsonl
  - split: test
    path: data/test_synthetic.jsonl
- config_name: proofwriter
  data_files:
  - split: test
    path: data/test_proofwriter.jsonl
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

This is the data of the CRUX research project (code: https://github.com/0xAnto/crux).
The project tests if rule-conditioned computation and counterfactual learning help a small encoder (about 17M parameters) to decide tasks with rules that it did not see in training.
The trained models are at https://huggingface.co/0xanto/crux.

## Usage

```python
from datasets import load_dataset

crux = load_dataset("0xanto/crux-data", "crux")  # train, dev_id, validation, test (synthetic)
proofwriter = load_dataset("0xanto/crux-data", "proofwriter", split="test")
boardgameqa = load_dataset("0xanto/crux-data", "boardgameqa", split="test")
dynabench = load_dataset("0xanto/crux-data", "dynabench", split="test")
print(crux["validation"][0]["options"])
```

## Record

Each record is one decision. All configs use the same top-level fields. The `meta` keys differ between the configs.

| Field | Type | Content |
|---|---|---|
| `id` | string | Unique in its file |
| `source` | string | The source data set, for example `synthetic` or `proofwriter` |
| `task_type` | string | `binary` (True / False / Unknown) or `multi_choice` |
| `state` | string | The facts |
| `instruction` | string | The rules, then the question after `Question:` |
| `options` | list of strings | The candidate answers |
| `target` | int | The index of the correct option |
| `meta` | dict | Source data. Synthetic: the template, the rule composition, the counterfactual pair (`pair_id`, `pair_role`, `edit_type`, `label_flip`), and the paraphrase group |

`docs/schema.md` and `docs/generator_spec.md` in the code repository define the record and the generator.

## Configs and splits

| Config | Split | Records | Content | License |
|---|---|---|---|---|
| `crux` | `train` | 75,646 | Synthetic counterfactual data from the project generator | Apache-2.0 |
| `crux` | `dev_id` | 4,162 | Synthetic, in-distribution development split (seen rules) | Apache-2.0 |
| `crux` | `validation` | 10,106 | Synthetic, unseen rule compositions. The project uses it for all model selection | Apache-2.0 |
| `crux` | `test` | 10,086 | Synthetic, unseen rule compositions | Apache-2.0 |
| `proofwriter` | `test` | 32,135 | ProofWriter open-world sets: depth5 18,857, birds_electricity 5,270, pararules 8,008 (`meta.subset`) | CC BY 4.0 |
| `boardgameqa` | `test` | 15,000 | BoardgameQA, 15 subsets of 1,000 (`meta.subset`) | CC BY |
| `dynabench` | `test` | 543 | The DynaBench test set of the DynaGuard paper | MIT |

The project evaluated the test data one time, in Step 8. Do not use the test data for model selection.
Super-NaturalInstructions has no config: `data/benchmarks/supernat_ids.jsonl` holds the record IDs, the task IDs, and the SHA-256 of each record only (4,100 records, 41 tasks), and `supernat_tasks.json` holds the license of each task. To get the records, download them from the official source and match them with the hashes.

## Integrity

`splits_manifest.json` gives the SHA-256 of each frozen split file of the project. The frozen test split is one file. Here it is two files, because the two sources have different `meta` keys. `MANIFEST.json` lists the two files in order (`test_files_in_order`). The two files joined in this order give the frozen test split, with the same SHA-256. `MANIFEST.json` also gives the SHA-256 of every file in this repository.

## Attribution and licenses

`NOTICE` gives the attribution of ProofWriter, RuleTaker, BoardgameQA, DynaBench, and Super-NaturalInstructions.
`DATA_LICENSES.md` records the license check of each source.

## Limitations

- The ProofWriter sets are the open-world versions (True / False / Unknown). Compare results only with published open-world results.
- The sentence frames and names of the ProofWriter depth5 set occur in the pretraining data of the Ettin encoders.
- The synthetic data comes from one generator, with fixed templates and vocabularies.
- DynaBench has 543 records, so the confidence intervals on it are wide.

## Citation

```bibtex
@misc{ranjith2026crux,
  author       = {Antony Ranjith F.},
  title        = {Does Rule Conditioning Help a 17M Encoder Decide on Unseen Rules? A Negative Result for the {CRUX} Architectures},
  year         = {2026},
  howpublished = {\url{https://github.com/0xAnto/crux}}
}
```
