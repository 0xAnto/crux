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
| Clark et al. 2020, Table 6 | BERT (the paper does not give the size) | RuleTaker depth-5 train | test | 96.9 |
| Our result (Colab T4, run `ettin17m_rt_d5_lr3e-5_ep10`) | Ettin-17M | RuleTaker depth-5 train | dev | 85.6 |

A realistic reference for a small encoder is the BERT row: 96.9 (Clark et al. 2020, arXiv:2002.05867, Table 6, row "Test (own)", column "MMax"). We found no published result for a smaller encoder on the depth-5 data. Pirozelli et al. (2023, arXiv:2312.11720, Table 2) report DistilBERT 86.55 and RoBERTa-base 98.04, but on the ParaRules version of RuleTaker, not on depth-5. Ettin-17M has 7 layers and a hidden size of 256. It is much smaller than these models.

## Colab T4 runs

All runs use seed 13, batch size 32, fp16, warmup ratio 0.06, linear decay, weight decay 0.01, `max_grad_norm` 1.0 (the Trainer default), and mean pooling (the `classifier_pooling` value of the Ettin config). The table gives the settings that differ from the first run. The run keeps the checkpoint with the best dev accuracy. Train time is `train_runtime` from `log_history.json`. It does not include the data download and the model load.

The settings are chosen on RuleTaker dev. Dev is a validation split. The dev accuracy is therefore not an unbiased estimate of the test accuracy.

| Run | Changed settings | Dev accuracy (higher is better) | Dev loss (lower is better) | Steps | Train time, s (lower is better) |
|---|---|---|---|---|---|
| `ettin17m_ruletaker_depth5_cwa` (first run, failed) | none: lr 1e-4, 3 epochs, eval every 1000 steps | 0.5768 | 0.6674 | 6543 | 417.9 |
| `ettin17m_rt_d5_lr3e-5_ep10` (best) | lr 3e-5, 10 epochs, eval every 2000 steps | 0.8556 | 0.4360 | 21810 | 1304.9 |
| `ettin17m_rt_d5_lr3e-5_ep3` (ablation: only the lr changes) | lr 3e-5 | 0.7709 | 0.4735 | 6543 | 415.8 |

Dev accuracy at each evaluation (higher is better):

- First run (eval at steps 1000 to 6543): 0.500, 0.555, 0.539, 0.571, 0.574, 0.570, 0.577.
- `ettin17m_rt_d5_lr3e-5_ep3` (eval at steps 1000 to 6543): 0.5660, 0.6631, 0.6644, 0.7424, 0.7553, 0.7679, 0.7709.
- `ettin17m_rt_d5_lr3e-5_ep10` (eval at steps 2000 to 21810): 0.5979, 0.7149, 0.7573, 0.8027, 0.8199, 0.8360, 0.8507, 0.8505, 0.8491, 0.8556, 0.8526. The dev loss is lowest at step 14000 (0.3376) and then rises. The train loss falls to about 0.18 to 0.20 near epoch 8. The model starts to overfit after about epoch 6.

`configs/step1_ruletaker.yaml` now holds the settings of the best run.

## Cause of the failed first run

The cause is the learning rate. With lr 1e-4 the model stays near chance. With lr 3e-5 and no other change, dev accuracy goes from 0.5768 to 0.7709 in the same 6543 steps. More epochs (10) then give 0.8556.

These checks found no bug in the pipeline:

- Data: train has 34,881 True and 34,881 False labels. Dev has 5,034 True and 5,034 False labels. The counts (69,762 and 10,068) agree with the README of the zip. Decoded token ids give `[CLS] context [SEP] question [SEP]`. The context, the question, and the label agree with the raw JSON for the same ids. The label map is false=0, true=1.
- Overfit test (CPU, fp32, lr 1e-4, batch 32): on 64 random train examples, train accuracy is 1.000 after 20 epochs. On the first 64 train examples (4 theories with question pairs "X" and "not X"), train accuracy is 0.969 after 60 epochs and 1.000 after 75 epochs. The head, the pooling, the loss, and the attention mask can fit the data.
- fp16 is not the cause. A local fp32 run (Apple MPS, lr 1e-4) also stays at about 0.58 on 2000 dev examples after 1000 steps.
- The pretrained model is sound. The masked-LM head predicts " Paris" for "The capital of France is <mask>."
- Accuracy by question depth of the first run (local fp32 evaluation of `best_model/`): depth 0 0.549, depth 1 0.569, depth 5 0.624. The model did not even learn depth-0 questions, which only need a lookup in the context. For the best run the values are depth 0 0.983, depth 1 0.712, depth 5 0.888.

A possible reason for the sensitivity to the learning rate: the classifier reuses the pretrained masked-LM prediction head (`head.norm.weight` has a mean of 3.80 and a maximum of 12.37). At the start the logits are large, the first loss values are high (1.018 at step 100 in the first run), and the gradient norm before clipping is high (97 at step 100). We did not test this reason.

Published fine-tuning settings (for comparison): Ettin sweeps lr over {1e-5, 3e-5, 5e-5, 8e-5, 1e-4} and weight decay over {1e-6, 5e-6, 8e-6, 1e-5} on GLUE (arXiv:2507.11412, Appendix F). ModernBERT uses a linear schedule with 6% warmup (repo `yamls/finetuning/glue`) and picks lr 5e-5 or 8e-5 for the base model on GLUE (arXiv:2412.13663, Table 6). Ettin does not publish the chosen settings per model size. Clark et al. (2020) inherit the RoBERTa settings for RACE (lr 1e-5, warmup 0.06, RoBERTa paper arXiv:1907.11692, Table 10).

## Local check

A tiny CPU run (5 steps, 300 train examples, 200 dev examples) checked that the pipeline runs end to end and writes `resolved_config.yaml`, `log_history.json`, `metrics.json`, the checkpoints, and `best_model/`. The run output was deleted. The accuracy of that run has no meaning and is not a result.

## Outputs of a run

`runs/step1/<run_name>/` holds `resolved_config.yaml` (with the seed and the parameter counts), `log_history.json` (the Trainer logs), `metrics.json` (the final dev metrics), `checkpoints/`, and `best_model/`. Git ignores `runs/`.

## Open items

- The 3-class open-world set (True/False/Unknown) comes from the ProofWriter release, not from this zip. Its files have a different format. Set `data.labels` to `["false", "true", "unknown"]` and adapt the loader then.
- The learning rate (3e-5) and the number of epochs (10) come from the runs above. The batch size (32) and the weight decay (0.01) are choices of the author. Only one seed is used. Change a value only with a note here.
