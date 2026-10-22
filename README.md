# CRUX

CRUX is a research project about small decision models.
It tests this hypothesis: explicit rule-conditioned computation, together with counterfactual learning, helps a small encoder (about 17M parameters) decide on rules that it did not see in training, better than a standard transformer classifier.

**The result is negative.** No CRUX architecture is reliably better than the baseline.

- Paper: `paper/main.tex` (build: `cd paper && tectonic main.tex`)
- Weights (11 models, 3 seeds each): https://huggingface.co/0xanto/crux
- Data: https://huggingface.co/datasets/0xanto/crux-data

## Architectures

All models start from the Ettin-17M encoder (`jhu-clsp/ettin-encoder-17m`).

| Model | Content |
|---|---|
| A | Ettin-17M with a standard decision head (the baseline) |
| B | A with an instruction-conditioned low-rank hypernetwork (the rule compiler) |
| C | B trained on counterfactual pairs, with a paraphrase consistency loss |
| D (CRUX) | C with cacheable state and instruction representations (split encoding) and a permutation-equivariant set-attention block over the options |

Controls and ablations: A-param (A with the trainable parameters of B, without conditioning), B-abl (B with hypernetwork adapters in the top 2 encoder layers), A+CF and B+CF (A and B trained on the counterfactual pairs), C+cache and C+equivariant (C with one part of D each), and D-CF (D without the counterfactual pairs and the consistency loss).

## Results

All models have the same budget: 12000 steps of 32 records, 1 hyperparameter trial, and 3 seeds (13, 42, 2026).
Each difference to A has a 95% paired bootstrap CI over the records (10,000 resamples). A difference is reliable only if its CI excludes 0. The CI does not include the seed variance.
The test split was evaluated one time.

Accuracy, mean over 3 seeds (higher is better):

| Model | test synthetic (unseen rules) | BoardgameQA | DynaBench | Super-NaturalInstructions |
|---|---|---|---|---|
| majority baseline | 0.336 | 0.367 | 0.508 | 0.442 |
| A | 0.640 | 0.373 | 0.499 | 0.440 |
| B | 0.636 | 0.367 | 0.475 | 0.434 |
| C | 0.513 | 0.368 | 0.534 | 0.441 |
| D | 0.442 | 0.368 | 0.510 | 0.433 |

- On test synthetic, B equals A on unseen_rule_accuracy (−0.004, the CI includes 0). B is 0.009 lower on counterfactual_paired_accuracy (the CI is below 0).
- All other models are reliably worse than A on both metrics. D is 0.198 lower than A on unseen_rule_accuracy.
- The models do not beat the majority baseline on the external benchmarks by much: BoardgameQA 0.006 below to 0.006 above, Super-NaturalInstructions 0.009 below to 0.001 above, DynaBench (543 records) 0.041 below to 0.037 above.
- On the ProofWriter open-world sets, the models are at, near, or below the majority baseline. The seed spread is larger than most differences.
- A is also the fastest model: 9.53 ms p50 latency at batch size 1 on an NVIDIA L4 (lower is better). D needs 19.80 ms without a cache and 11.76 ms with the state and the instruction cached.

`steps.md` has the full record of each step: all tables, all CIs, the deviations from the plan, and the limitations.

## Deviations from the plan

- The external baselines (fine-tuned Ettin-32M and DeBERTa-v3-xsmall) are not evaluated. Their code and configs are in the repository.
- Step 9 (mobile optimization, INT8, Pixel 8) is skipped, because no architecture is better than A.
- `configs/protocol.yaml` fixed the seeds, the metrics, and the statistics before training. The training budget (12000 steps, 1 trial) was set later, after a 6000-step pilot.
- All runs used an NVIDIA L4 on Google Colab, not the T4 of the plan.

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

The code uses fp16 mixed precision and SDPA attention.
The notebooks in `notebooks/` run Steps 1, 2, 4, 5, 6, and 7 on Colab. `crux/colab_sweep.py` and `crux/colab_eval.py` run the training sweeps and the Step 8 benchmark on Colab VMs with the `colab` CLI.

## Load a released model

The models are not `transformers` models. They need the code of this repository.
The model card at https://huggingface.co/0xanto/crux has a complete script. In short:

```python
import yaml
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from crux.models import build_model

cfg = yaml.safe_load(open(hf_hub_download("0xanto/crux", "weights/A/s0/config.yaml")))
model = build_model(cfg["model"])
model.load_state_dict(load_file(hf_hub_download("0xanto/crux", "weights/A/s0/model.safetensors")))
```

## Reproduce the experiments

Run the commands from the repository root. Each step writes its outputs under `data/` or `runs/` (both are git-ignored).

| Step | Command |
|---|---|
| 1. Smoke test on RuleTaker | `python -m crux.train_step1 --config configs/step1_ruletaker.yaml` |
| 2. Convert the datasets | `python -m crux.data.ruletaker` (also `proofwriter`, `boardgameqa`, `dynabench`, `supernat`) |
| 3. Generate the synthetic data | `python -m crux.gen.generator`, then `python -m crux.gen.engine check data/processed/synthetic/main/all.jsonl` and `python -m crux.gen.audit data/processed/synthetic/main/` |
| 4. Build and verify the frozen splits | `python -m crux.splits build`, then `python -m crux.splits verify` |
| 5 to 7. Train one run | `python -m crux.train --config configs/step5.yaml --arch B --trial 0 --seed-index 0` |
| 5 to 7. Aggregate the runs | `python -m crux.aggregate runs/step5` |
| 5 to 7. Run a sweep on Colab | `python -m crux.colab_sweep --config configs/step5.yaml --root runs/step5` |
| 8. Benchmark all models | `python -m crux.colab_eval --commit <commit>`, then `python -m crux.aggregate --bench configs/step8_eval.yaml` and `python -m crux.plots` |
| 10. Build the Hub release | `python -m crux.release`, then `hf upload 0xanto/crux release/model .` and `hf upload 0xanto/crux-data release/dataset . --repo-type dataset` |

The architectures of Steps 5 to 7 are in `configs/step5.yaml` (A, B, B-abl, A-param), `configs/step6.yaml` (C, A+CF, B+CF), and `configs/step7.yaml` (D, C+cache, C+equivariant, D-CF).
`crux.splits build` stops if a split hash differs from `configs/splits_manifest.json`.
The test split opens only with `crux.benchmark --final`.
A rerun gives the reported numbers within the seed variance, not bit for bit.

## Repository layout

| Path | Content |
|---|---|
| `goal.md`, `steps.md` | The goal and the record of Steps 0 to 10 |
| `configs/` | `protocol.yaml`, the configs of each step, and `splits_manifest.json` (SHA-256 and counts of the frozen splits) |
| `crux/schema.py`, `crux/data/` | The decision record, the converters, and the batching code |
| `crux/gen/` | The synthetic generator, the independent rule engine, and the shortcut audit |
| `crux/splits.py`, `crux/contamination.py` | The split freezer, the leakage check, and the n-gram scan against the Ettin pretraining data |
| `crux/models.py`, `crux/train.py` | The architectures and the training |
| `crux/metrics.py`, `crux/aggregate.py`, `crux/benchmark.py`, `crux/plots.py` | The metrics, the bootstrap, the benchmark, and the plots |
| `crux/colab_sweep.py`, `crux/colab_eval.py` | The Colab drivers |
| `crux/release.py` | The builder of the Hugging Face release |
| `docs/` | The schema, the generator, the splits, the metrics, the contamination check, the literature review, and the Hub cards |
| `paper/` | The paper |
| `tests/` | The unit tests |

## License and attribution

The code, the trained weights, and the synthetic data have the Apache-2.0 license (`LICENSE`).
`NOTICE` gives the attributions of the third-party parts:

- The weights start from Ettin-17M (MIT).
- The data release contains records from ProofWriter (CC BY 4.0), RuleTaker (Apache-2.0), BoardgameQA (CC BY), and DynaBench (MIT).
- The release does not contain Super-NaturalInstructions instances. It has the task IDs and the record hashes only, because each task has its own instance license.

`DATA_LICENSES.md` records the license check of each source.

## Citation

```bibtex
@misc{ranjith2026crux,
  author       = {Antony Ranjith F.},
  title        = {Does Rule Conditioning Help a 17M Encoder Decide on Unseen Rules? A Negative Result for the {CRUX} Architectures},
  year         = {2026},
  howpublished = {\url{https://github.com/0xAnto/crux}}
}
```
