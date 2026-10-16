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

There are no model results yet.
The only measurement is the Step 1 pipeline smoke test: Ettin-17M reaches 0.856 accuracy on the RuleTaker depth-5 dev split.
This number checks the training pipeline. It is measured on dev, one seed, and it is not an official result.
The project makes no claim about B, C, or D.

| Step | Content | Status |
|---|---|---|
| 0 | Protocol, license checks, literature review | Complete |
| 1 | Baseline smoke test | Complete |
| 2 | Unified decision schema, data loaders, converters | Complete |
| 3 | Counterfactual dataset generator | Complete |
| 4 | Evaluation dataset and frozen splits | Complete |
| 5 | Architecture A (official) and B | Not started |
| 6 | Architecture C | Not started |
| 7 | Architecture D | Not started |
| 8 | Benchmark of all models | Not started |
| 9 | Mobile optimization | Not started |
| 10 | Paper and release | Not started |

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
- `docs/`: `schema.md`, `generator_spec.md`, `splits.md`, `metrics.md`, `step1_report.md`, `contamination.md`, `literature_review.md`.
- `notebooks/`: one Colab notebook per step.
- `tests/`: unit tests.
- `DATA_LICENSES.md`: licenses of all datasets and model weights.

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Training targets Google Colab with a T4 GPU. The T4 does not support bf16 or FlashAttention 2, so the code uses fp16 mixed precision and SDPA attention.
`notebooks/` has one notebook per step. Each notebook runs on Colab.

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

Run the tests.

```
pytest
```

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
The repository does not redistribute processed data. The `data/` directory is git-ignored. The converters build the data from the official sources.
BoardgameQA, DynaBench, and Super-NaturalInstructions are for evaluation only. The project does not train on them.
Super-NaturalInstructions instances are not redistributed. The release will include task IDs and hashes only.

## License

The code is released under the Apache-2.0 license. See `LICENSE`.
Attribution:

- ProofWriter is licensed under CC BY 4.0. Cite the source when you use it.
- Ettin (`jhu-clsp/ettin-encoder-17m`) is licensed under the MIT license. Keep its copyright notice.
