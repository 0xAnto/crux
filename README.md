# CRUX

CRUX is a compact System 1 decision model with about 17M to 30M parameters.
It is an open research project.
The research hypothesis: explicit rule-conditioned computation, together with counterfactual learning, helps a small neural network generalize to unfamiliar decision tasks better than a conventional transformer classifier.
The project compares four architectures under one pre-registered protocol:

- **A, Baseline:** Ettin-17M encoder with a standard decision head.
- **B, Rule Compiler:** A with an instruction-conditioned low-rank hypernetwork.
- **C, Counterfactual Learning:** B with counterfactual training and a consistency loss.
- **D, CRUX:** C with independently cacheable state and rule representations and permutation-equivariant candidate scoring.

Controls: A-param (A with the same trainable parameters as B, without conditioning), A+CF, B+CF, and two ablations of D (C+cache, C+equivariant). The external baselines are fine-tuned Ettin-32M and DeBERTa-v3-xsmall. See `goal.md` and `steps.md`.

## Status

Steps 0 to 8 are complete. Step 9 (mobile optimization) is skipped. Step 10 (paper and release) is in progress.

| Step | Content | Status |
|---|---|---|
| 0 | Protocol, license checks, literature review | Complete |
| 1 | Baseline smoke test | Complete |
| 2 | Unified decision schema, data loaders, converters | Complete |
| 3 | Counterfactual dataset generator | Complete |
| 4 | Evaluation dataset and frozen splits | Complete |
| 5 | Architecture A (official) and B | Complete |
| 6 | Architecture C | Complete |
| 7 | Architecture D | Complete |
| 8 | Benchmark of all models | Complete (the external baselines are not evaluated) |
| 9 | Mobile optimization | Skipped: no architecture is better than A, so a mobile benchmark of A measures no CRUX part |
| 10 | Paper and release | In progress |

## Results

The result is negative. The hypothesis is not supported.
The project read the frozen test split one time. Each value is the mean over 3 seeds. The 95% CI covers the records, not the seeds.

- No architecture is reliably better than A on the test synthetic subset (unseen rules). B equals A (difference -0.004, the CI includes 0).
- All other models are reliably worse than A. D is 0.198 lower than A.
- On counterfactual paired accuracy, all models are reliably lower than A.
- No model is materially above the majority baseline on BoardgameQA, DynaBench, or Super-NaturalInstructions.
- The models do not transfer from the synthetic data to the ProofWriter open-world sets. The seed spread is larger than most differences there.

Accuracy (higher is better):

| Model | test synthetic | BoardgameQA | DynaBench | Super-NaturalInstructions |
|---|---|---|---|---|
| majority baseline | 0.336 | 0.367 | 0.508 | 0.442 |
| A | 0.640 | 0.373 | 0.499 | 0.440 |
| B | 0.636 | 0.367 | 0.475 | 0.434 |
| C | 0.513 | 0.368 | 0.534 | 0.441 |
| D | 0.442 | 0.368 | 0.510 | 0.433 |

The other models, the ProofWriter sets, the calibration, and the limitations are in `steps.md` (Step 8) and in `docs/model_card.md`.
The external baselines (Ettin-32M, DeBERTa-v3-xsmall) are not evaluated.

## Repository layout

- `goal.md`: project goal, hypothesis, and success criterion.
- `steps.md`: Steps 0 to 10, with the status of each step.
- `configs/`: `protocol.yaml` (pre-registered protocol), the settings of Steps 1 to 4, and `splits_manifest.json` (the SHA-256 and counts of the frozen splits).
- `crux/schema.py`: the decision record schema and its validator.
- `crux/data/`: loaders and converters (RuleTaker, ProofWriter, BoardgameQA, DynaBench, Super-NaturalInstructions) and the batching code.
- `crux/train_step1.py`: the Step 1 training script.
- `crux/gen/`: the Step 3 generator, the independent rule engine, and the shortcut audit.
- `crux/splits.py`: the Step 4 split freezer, the leakage check, and `load_split`.
- `crux/metrics.py`: the evaluation metrics (`docs/metrics.md`).
- `crux/contamination.py`: the n-gram overlap scan of the test split against the Ettin `tulu_flan` data.
- `crux/train.py`, `crux/aggregate.py`, `crux/benchmark.py`, `crux/plots.py`: Steps 5 to 8 (training, selection and bootstrap, the benchmark, the plots).
- `crux/colab_sweep.py`, `crux/colab_eval.py`: the Colab drivers for the training sweeps and the Step 8 benchmark.
- `crux/release.py`: the builder of the local release folder.
- `docs/`: `schema.md`, `generator_spec.md`, `splits.md`, `metrics.md`, `step1_report.md`, `contamination.md`, `literature_review.md`.
- `notebooks/`: one Colab notebook per step.
- `tests/`: unit tests.
- `DATA_LICENSES.md`: licenses of all datasets and model weights.
- `NOTICE`: the attributions that the third-party licenses require.

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Training targets Google Colab with a T4 GPU. The T4 does not support bf16 or FlashAttention 2, so the code uses fp16 mixed precision and SDPA attention.
`notebooks/` has one notebook per step. Each notebook runs on Colab.
A clean Colab run gives the reported numbers only within the seed variance.

## Usage

Run the commands from the repository root.

Step 1: train Ettin-17M on RuleTaker.

```
python -m crux.train_step1 --config configs/step1_ruletaker.yaml
```

Step 2: convert the datasets. Each converter writes JSONL files under `data/processed/`.

```
python -m crux.data.ruletaker
python -m crux.data.proofwriter
python -m crux.data.boardgameqa
python -m crux.data.dynabench
python -m crux.data.supernat
```

Step 2: validate a processed file, and show its batches.

```
python -m crux.schema data/processed/ruletaker/depth3/train.jsonl
python -m crux.data.batching data/processed/ruletaker/depth3/dev.jsonl
```

Step 3: run the generator, check its labels with the independent rule engine, and audit the output for shortcuts.
The generator writes `all.jsonl`, `theories.jsonl`, and `report.json` to `data/processed/synthetic/main/`.

```
python -m crux.gen.generator [--config configs/step3_generator.yaml] [--seed N] [--total N] [--out DIR]
python -m crux.gen.engine check data/processed/synthetic/main/all.jsonl
python -m crux.gen.audit data/processed/synthetic/main/
```

Step 4: build and freeze the splits, then check the files against the manifest.
The build reads the Step 2 and Step 3 outputs and writes `data/splits/`. It stops if a hash differs from the frozen manifest.

```
python -m crux.splits build
python -m crux.splits verify
```

Steps 5 to 7: train one model (one architecture, one trial, one seed index). The notebooks `step5_train.ipynb`, `step6_train.ipynb`, and `step7_train.ipynb` loop over all runs.
The architectures are in `configs/step5.yaml` (A, B, B-abl, A-param), `configs/step6.yaml` (C, A+CF, B+CF), and `configs/step7.yaml` (D, C+cache, C+equivariant, D-CF).
The aggregation selects the trial and writes the mean, the std, and the paired bootstrap CI.

```
python -m crux.train --config configs/step5.yaml --arch B --trial 0 --seed-index 0
python -m crux.aggregate runs/step5
```

Colab drivers: `crux.colab_sweep` runs a sweep on Colab L4 VMs with the `colab` CLI. `crux.colab_eval` runs the Step 8 benchmark.
Both drivers are restart-safe. The VMs check out a pinned commit.

```
python -m crux.colab_sweep --config configs/step5.yaml --root runs/step5
python -m crux.colab_eval --commit <commit>
```

Step 8: `crux.colab_eval` calls `crux.benchmark --final` on the VMs. The test split needs `--final`, and the project reads it one time.
The report and the plots come from the downloaded outputs.

```
python -m crux.benchmark --final --baselines
python -m crux.aggregate --bench configs/step8_eval.yaml
python -m crux.plots
```

Step 10: build the release folders `release/model` and `release/dataset`. The builder uploads nothing. The two `hf upload` commands publish them.

```
python -m crux.release
hf upload 0xanto/crux release/model .
hf upload 0xanto/crux-data release/dataset . --repo-type dataset
```

Run the tests.

```
pytest
```

## Release contents

The weights and the data are on the Hugging Face Hub. They are not in git.

- Model repo https://huggingface.co/0xanto/crux: `weights/<model>/s<k>/model.safetensors` and `config.yaml` for 11 models, 3 seeds each. The file holds the model state dict only. `README.md` is the model card (`docs/model_card.md`).
- Dataset repo https://huggingface.co/datasets/0xanto/crux-data: the synthetic splits, `test.jsonl` (synthetic and ProofWriter records), BoardgameQA, and DynaBench. For Super-NaturalInstructions, it holds the task IDs, the per-task licenses, and the record hashes only. It holds no instance. `README.md` is the dataset card (`docs/dataset_card.md`).
- Both repos hold `NOTICE`, `LICENSE`, `DATA_LICENSES.md`, and `MANIFEST.json` (SHA-256 of every file, the split hashes, and the git commit).

## Protocol

`configs/protocol.yaml` pre-registers the protocol before any training.
It fixes these rules:

- 3 seeds (13, 42, 2026) for every main configuration and every control.
- Equal training budget, equal data, and equal hyperparameter search budget for all models.
- All model selection uses the validation split.
- The test split is frozen in Step 4. The project reads it once, in Step 8.
- Each difference to A is reported with a 95% paired bootstrap CI (10,000 resamples). An improvement is reliable only if the CI excludes 0.
- Each result reports the total and the trainable parameter count.

## Data and licenses

`DATA_LICENSES.md` records the license of each dataset and each model weight, with the official source.
The git repository holds no processed data. The `data/` directory is git-ignored. The converters build the data from the official sources. The dataset repo on the Hugging Face Hub holds the processed data that the licenses permit.
BoardgameQA, DynaBench, and Super-NaturalInstructions are for evaluation only. The project does not train on them.
Super-NaturalInstructions instances are not redistributed. The release includes task IDs and hashes only.

## License and attribution

The code, the trained weights, and the synthetic counterfactual dataset are released under the Apache-2.0 license. See `LICENSE`.
`NOTICE` has the full attributions. In short:

- ProofWriter is licensed under CC BY 4.0. The release contains records derived from it. Cite Tafjord et al. (2021).
- Ettin (`jhu-clsp/ettin-encoder-17m`) is licensed under the MIT license. The weights start from it. `NOTICE` keeps its copyright notice.
- BoardgameQA (CC BY) and DynaBench (MIT) records are in the release with attribution.
- Super-NaturalInstructions instances are not in the release. Each task has its own instance license (`DATA_LICENSES.md`).
