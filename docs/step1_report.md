# Step 1 report: baseline smoke test

Step 1 checks the training pipeline. It does not give an official result for Architecture A. The official A results come from Step 5, on the frozen splits.

## Setup

| Item | Value |
|---|---|
| Model | `jhu-clsp/ettin-encoder-17m` (ModernBERT architecture, 16.86M parameters with a 2-class head) |
| Code | `crux/train_step1.py`, `crux/data/ruletaker.py` |
| Run settings | `configs/step1_ruletaker.yaml` |
| Training | Hugging Face `Trainer`, `AutoModelForSequenceClassification`, SDPA attention, fp16 on CUDA |
| Seed | 13 (the first seed of `configs/protocol.yaml`). Step 1 uses one seed only. |
| Dataset | RuleTaker `original/depth-5` (closed world, labels True/False). This is the "DMax" set of Clark et al. (2020). |
| Dataset source | https://aristo-data-public.s3-us-west-2.amazonaws.com/ruletaker/rule-reasoning-dataset-V2020.2.5.zip (linked from the official repo https://github.com/allenai/ruletaker). The zip SHA-256 is in the config. |
| Dataset license | Apache-2.0. See `DATA_LICENSES.md`. |
| Input | Sentence pair: `[CLS] context [SEP] question [SEP]`. The context holds the facts and the rules. The code truncates the context only, at 512 tokens. `metrics.json` reports the number of truncated examples. |
| Transformers version | ModernBERT needs transformers 4.48 or later. The local check used 5.19. |

The depth-5 files hold 69,762 train questions, 10,068 dev questions, and 20,192 test questions (README of the zip).

## Splits and the test split

- The run trains on the RuleTaker train split.
- The run selects the best checkpoint on the RuleTaker dev split, and it reports dev accuracy.
- The code does not read the RuleTaker test split. Step 4 may put this split in the frozen test split, and the protocol forbids test reads outside Step 8.
- The zip file contains the test files. The download gets the whole zip. The loader extracts only `train.jsonl` and `dev.jsonl`, and `load_split` raises an error for any other split.

## Published reference

Source: Clark, Tafjord, Richardson. "Transformers as Soft Reasoners over Language." IJCAI 2020. arXiv:2002.05867 (v2), Table 1, row "Test (own)", column "MMax". The model is RoBERTa-large, fine-tuned on RACE first. It trains on the depth-5 data (DMax) and tests on the DMax test split.

Our number is on the **dev** split. The published number is on the **test** split. The two numbers are not a strict like-for-like comparison. The models also differ (Ettin-17M against RoBERTa-large), so a gap is not an error of the pipeline.

Accuracy of the Step 1 run (higher is better):

| Run | Model | Training data | Evaluation split | Accuracy % (higher is better) |
|---|---|---|---|---|
| Published (Clark et al. 2020, Table 1) | RoBERTa-large (+RACE) | RuleTaker depth-5 train | test | 99.2 |
| Our result (Colab T4) | Ettin-17M | RuleTaker depth-5 train | dev | |

Fill the "Our result (Colab T4)" row after the Colab run of `notebooks/step1_smoke.ipynb`. Read the value from `runs/step1/ettin17m_ruletaker_depth5_cwa/metrics.json` (`dev_accuracy`). Multiply by 100.

## Local check

A tiny CPU run (5 steps, 300 train examples, 200 dev examples) checked that the pipeline runs end to end and writes `resolved_config.yaml`, `log_history.json`, `metrics.json`, the checkpoints, and `best_model/`. The run output was deleted. The accuracy of that run has no meaning and is not a result.

## Outputs of a run

`runs/step1/<run_name>/` holds `resolved_config.yaml` (with the seed and the parameter counts), `log_history.json` (the Trainer logs), `metrics.json` (the final dev metrics), `checkpoints/`, and `best_model/`. Git ignores `runs/`.

## Open items

- The 3-class open-world set (True/False/Unknown) comes from the ProofWriter release, not from this zip. Its files have a different format. Set `data.labels` to `["false", "true", "unknown"]` and adapt the loader then.
- The learning rate, the batch size, and the number of epochs are choices of the author. They are untested on a GPU. Change them only with a note here.
