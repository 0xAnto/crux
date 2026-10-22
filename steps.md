# CRUX: Execution Steps

Do the steps in sequence. Do not start a step before the previous step is complete.

## Common instruction (prepend to every prompt)

> Build this as part of the CRUX research project. Use Python, PyTorch, Hugging Face, and Google Colab (T4 GPU). Keep the code modular, reproducible, memory-efficient, and compatible with the previous steps. Include unit tests, checkpoints, logging, and a runnable Colab notebook. Read all splits, seeds, and criteria from `configs/protocol.yaml` (Step 0). Use fp16 mixed precision and SDPA attention, because the T4 does not support bf16 or FlashAttention 2. Use the validation split for all model selection. Do not read the test split outside Step 8. Do not fabricate benchmark results.

## Protocol rules (apply to all steps)

- **Equal budget:** All architectures and controls get the same number of training steps, the same data, and the same hyperparameter search budget (for example, 6 trials on the validation split).
- **Seeds:** Train every main configuration and every control with 3 seeds. Do not select configurations before you run the seeds.
- **Test split:** Freeze the test split in Step 4. Evaluate on it one time, in Step 8.
- **Statistics:** Report the mean ± standard deviation across seeds and a 95% paired bootstrap CI for each difference between models.
- **Parameters:** Report the total and the trainable parameter count with every result.

---

## Step 0: Protocol and Pre-registration

- [x] Write `configs/protocol.yaml` with the seeds, the split definitions, the metrics, and the statistical test. Freeze it before Step 1.
- [x] Do not set numeric targets now. There are no measurements yet. Use these open criteria:
  - [x] An improvement is "statistically reliable" only if the 95% paired bootstrap CI of the difference to A excludes 0 on all 3 seeds together.
  - [x] Report every effect size with its CI. Do not call an effect "large" or "small" without a reference value.
  - [x] Mobile: measure and report model size, peak RAM, p50/p95 latency, and the INT8 accuracy loss. Name the Android device now, so that all measurements use the same device.
  - [x] Optional: after Step 5 measures A, you can add numeric targets for B, C, and D. Record the date of the change in `protocol.yaml` and in the paper.
- [x] Check the license of each dataset (RuleTaker, ProofWriter, and others) on the official source. Record the results in `DATA_LICENSES.md`. Do not use a mirror license label as the source.
- [x] Write a short literature review (instruction-conditioned hypernetworks such as HyperTuning and HINT; counterfactual consistency training; rule reasoning with RuleTaker and ProofWriter). Record which CRUX parts are novel and which parts are not novel.
- [x] Name the external baselines: fine-tuned Ettin-32M and fine-tuned DeBERTa-v3-xsmall. Jev has no public weights. Compare against Jev only if you get API access, only on public tasks, and label the result "API, not reproducible".
- [x] Use these external benchmarks for Jev-type decisions. Use them for evaluation only. Do not train on them, except in a separate and labeled fine-tune experiment.
  - [x] **BoardgameQA** (Google, NeurIPS 2023): facts, rules, exceptions, and rule priorities → proved / disproved / unknown. It tests the same rule types as the Step 3 generator, but with text from a different source.
  - [x] **DynaBench** (DynaGuard paper, 2025): a user-defined policy plus a dialogue → compliant / not compliant. The test policies are longer than the training policies and are not in the training data. This is the closest public match to Jev guardrail decisions. The test set has only 543 examples, so the CIs are wide.
  - [x] **Super-NaturalInstructions**, classification tasks only, official held-out task split: an instruction plus options → one option. It tests unseen instructions and a variable number of options.
  - [x] Confirm the license of each benchmark from its official repository and record it in `DATA_LICENSES.md`. If a license does not permit research use, remove that benchmark.
  - [x] Do not use ShARC: it has no known license, and published work reports spurious patterns in its labels.

**Done (2026-10-14, commit `21be4c8`):**
- `configs/protocol.yaml` is frozen. Seeds: 13, 42, 2026. Bootstrap: 10000 resamples. "Reliable" means the 95% CI pooled over the 3 seed pairs excludes 0.
- The mobile device is the Google Pixel 8 (Tensor G3, 8 GB RAM).
- `DATA_LICENSES.md`: no benchmark is removed. The 16 Super-NaturalInstructions held-out tasks with license "Unknown" are excluded.
- `docs/literature_review.md`: no CRUX part is novel alone. Only the combination at 17–30M parameters can be novel.
- Project license: Apache-2.0 for the code, the weights, and the synthetic data. CC BY 4.0 for the paper, only if the venue requires it.

## Step 1: Baseline Smoke Test

- [x] Train Ettin-17M (`jhu-clsp/ettin-encoder-17m`) on RuleTaker with fp16 mixed precision.
- [x] Use a head that matches the labels: 2 classes for closed-world data, 3 classes (True/False/Unknown) for open-world data.
- [x] Compare the accuracy with the published RuleTaker results. This step only checks the training pipeline. The official A results come from Step 5, on the frozen splits.
- [x] Save the checkpoints, the configurations, and the training metrics.
- [x] Check for contamination: search the Ettin pretraining data documentation for RuleTaker and ProofWriter. Record the result.

**Done (2026-10-15):**
- Contamination check: `docs/contamination.md`. RuleTaker and ProofWriter are not in the Ettin documentation. The Flan data in Ettin can contain the RuleTaker tasks task966 and task967 and 115 of the 119 Super-NaturalInstructions held-out tasks. This is not confirmed.
- Step 1 trains and evaluates on RuleTaker dev only. It does not read RuleTaker test.
- Pipeline: `crux/train_step1.py`, `crux/data/ruletaker.py`, `configs/step1_ruletaker.yaml`, `notebooks/step1_smoke.ipynb`. Data: RuleTaker `original/depth-5` (closed world, 2 classes) from the official Ai2 zip. The config pins the SHA-256 of the zip. The loader refuses the test split.
- The CPU smoke run (5 steps) passes end to end. Its output is deleted. It is not a result.
- Colab T4 runs (dev accuracy, higher is better): lr 1e-4, 3 epochs: 0.577 (near chance). lr 3e-5, 3 epochs: 0.771. lr 3e-5, 10 epochs: **0.856** (best, 1305 s). Cause: the lr 1e-4 is too high. No bug: the labels are 50/50, the inputs match the raw JSON, and the model overfits 64 examples. Details: `docs/step1_report.md`.
- Reference: BERT 96.9 and RoBERTa-large 99.2 on depth-5 test (Clark et al. 2020, Table 6). Our number is on dev, and dev also selects the checkpoint, so it is slightly optimistic. The pipeline check passes.

## Step 2: Unified Decision Schema

- [x] Build a preprocessing pipeline that converts each dataset into `{state, instruction, options, target}` for the binary, multi-choice, and scoring task types.
- [x] Keep `state` (facts and evidence) and `instruction` (rules and question) as separate fields. Architecture D encodes and caches these fields independently.
- [x] Support a variable number of options per example.
- [x] Implement tokenization, dynamic batching, dataset validation, and reusable PyTorch DataLoaders.

**Done (2026-10-15):**
- Schema `{id, source, task_type, state, instruction, options, target, meta}` in `crux/schema.py`. Binary tasks use the candidate form. One label order in all sources: positive, negative, "Unknown". Details: `docs/schema.md`.
- `crux/data/batching.py`: joint mode (A/B/C) and separate mode (D). Truncation cuts the rules and keeps the "Question: ..." tail. Training uses `FixedBatchSampler` (32 records per batch), so equal steps give equal data in all modes. Evaluation uses `TokenBudgetSampler`. Both samplers are for one process only.
- Converters (records): RuleTaker depth 3: train 70,108, dev 9,944. ProofWriter OWA depth 3: train 69,388, dev 10,302. Evaluation only: BoardgameQA 37,500 (15 subsets), DynaBench 543, Super-NaturalInstructions 142,350 (41 tasks). Test splits of RuleTaker and ProofWriter are not opened.
- A critical review found 9 defects (label order, question loss in truncation, budget, depth-5 use, and others). All are fixed. 51 tests pass.
- Colab notebook: `notebooks/step2_data.ipynb`.

## Step 3: Counterfactual Dataset Generator

- [x] Build a deterministic synthetic generator that supports logical rules, negations, thresholds, exceptions, priorities, and multi-step dependencies.
- [x] Give each example a template ID, a composition ID, and a vocabulary ID. Step 4 uses these IDs to make the splits.
- [x] Generate 100K labeled examples with matched counterfactual pairs. A pair changes one fact or one rule, and the label changes or stays the same as the rule engine computes.
- [x] Include multi-choice examples with 2 to 8 options. Without these, Step 7 cannot measure the effect of candidate scoring.
- [x] Generate paraphrases of the instructions that are logically equivalent.
- [x] Validate every label with an executable symbolic rule engine. Reject an example if the engine and the generator disagree.

**Done (2026-10-15):**
- Spec: `docs/generator_spec.md`. Open world, classical negation, exceptions block, pairwise priorities. 32 compositions of 5 rule types (`conj`, `neg`, `thr`, `exc`, `pri`). 7 are held out, stratified by the number of rule types: val c05, c08, c11, c23; test c18, c28, c31. 15 templates, 12 vocabulary pools. Block partition 0.8 / 0.1 / 0.1.
- Output: `data/processed/synthetic/main/` has 100,000 records (train 79,808, val 10,106, test 10,086), 66,583 theories, and 33,292 units (1 of 33,293 planned units failed). 69,930 records are binary and 30,070 are multi-choice (2 to 8 options). The flip share of the pairs is 0.600. The fill rate of each cell is 0.988 or more. Two runs give the same SHA-256 (`all.jsonl` da6534cc…, `theories.jsonl` 19f65d0e…).
- The rule engine (`crux/gen/engine.py`) and the generator (`crux/gen/generator.py`) are independent implementations. They agree on all 100K records (0 disagreements) and on 70K random fuzz theories.
- `crux/gen/audit.py` checks shortcuts and invariants against fixed thresholds. All 41 checks pass on the final data. All 310 tests pass.
- Fact construction: after the fact search, a Metropolis-Hastings walk draws again the facts of the queried entity that a condition reads (section 9.5). Without it, the facts on the query path depended on the label in a different way in each composition, and a fact-count probe trained on the train block scored 0.29 on the val block (chance 0.33). The free facts are random (an earlier fact controller hid an off-path cue and did not remove it).
- Changes of the spec targets after the first 100K runs (reasons in `docs/generator_spec.md`, section 9.2):
  - The binary label target ± 0.03 applies to the orig records at depth 1 or more (train: 0.333 / 0.334 / 0.333). All records have ± 0.05 (train: 0.321 / 0.323 / 0.357 Unknown). A depth-0 Unknown cannot flip, and 7 compositions have no True↔False flip. Balancing the totals would make the orig label show the flip status.
  - `exception_remove` and `priority_swap` have the target ≥ 0.005 of the pairs (realized 0.010 and 0.011). The other types keep ≥ 0.05.
  - `max_attempts` is 200 (was 100). With the walk, c11 at depth 5 (binary) filled only 0.94 of its cell.
- Changes of the audit:
  - The lower bound of the model checks uses the chance agreement of the predictions, not the shuffled-label null (`label_cue.bow.ood`: acc 0.321, null 0.367, chance 0.327).
  - The label probes split the facts of the queried entity at the query path. `label_cue.path` and `label_cue.ref` compare the data with a label-blind reference (the facts on the path drawn at random and labeled again), for the orig and the edit records, with an upper limit (+0.03) and a lower limit (chance − 0.03).
  - The count and text probes shuffle the null labels within each composition. The plain count probe runs on the orig records only. The plain probes `label_cue.counts` (all records) and `label_cue.counts_edit` are removed: an edit forces a label signal in the structure (a removed rule moves its facts off the path), so the edit records are compared with the reference.
- Multi-choice pairs with a rule edit: train 631 of 7,954 units (7.9%), val 31 of 1,029 (all in c23), test 157 of 1,018 (c18, c28, c31). They need `pri` (section 9.5).
- Rare edit types (units, all tasks): `exception_remove` train 255, val 22, test 43; `priority_swap` train 327, val 4, test 37.
- Reviews: round 1 fixed inert stays, a polarity cue, wrong pair feasibility, and easy distractors, and added rule-side edits. Round 2 found a stay marker and a height cue in multi-choice. The audit then found more cues. Round 3 (after the walk): approved with nits, no wrong labels and no hidden shortcut. Its fixes: the path check for edit records, the worst val composition in the audit detail, the claim of the walk, and a unit test for the walk.

## Step 4: Evaluation Dataset and Frozen Splits

- [x] Make train, validation, and test splits. The test split contains rule templates, compositions, vocabulary, and paraphrases that do not occur in the train split.
- [x] Hold out by reasoning depth and by domain: use ProofWriter depth 5, Birds-Electricity, and ParaRules only for tests.
- [x] Check for leakage automatically: no shared template IDs, no near-duplicate texts (n-gram overlap) between train and test.
- [x] Record a hash for each split file, then freeze the splits.
- [x] Implement these metrics: accuracy, counterfactual paired accuracy, paraphrase consistency, calibration (ECE), option-order invariance, and unseen-rule accuracy.

**Done (2026-10-16):**
- Splits (`crux/splits.py`, `configs/step4_splits.yaml`, `docs/splits.md`). Frozen in `configs/splits_manifest.json` and `configs/protocol.yaml`. Records: train 75,646, dev_id 4,162, validation 10,106, test 42,221 (synthetic test block 10,086, ProofWriter OWA depth 5 18,857, Birds-Electricity 5,270, ParaRules 8,008). Evaluation-only benchmarks: BoardgameQA 15,000 (the official test file of each of the 15 subsets), DynaBench 543, Super-NaturalInstructions 4,100.
- SHA-256: train `bd8cc114…`, validation `d414891b…`, test `fbe0cfca…`. Two builds give the same hashes. `build` writes to `data/splits.new/` and replaces the frozen files only with `--refreeze`. `load_split` checks the hash against the manifest and `protocol.yaml`, and refuses `test` without `allow_test=True` (Step 8 only).
- dev_id: 5% of the train-block units (seeded hash of `unit_id`). It is an in-distribution dev set with the train templates, compositions, and vocabulary. It is not for model selection.
- Test-only converters: `to_test_records` in `crux/data/proofwriter.py` reads the OWA `meta-test.jsonl` of depth 5, Birds-Electricity, and NatLang (= ParaRules), with 3 classes as the synthetic data. The ParaRules `theory` is a paraphrase, so the facts and the rules come from `sentences` and `mappings`. All 33,308 records agree with the raw labels (review check).
- Leakage: no shared id (template, template group, composition, vocabulary, paraphrase group, unit, theory hash) between train, validation, and test, and dev_id against validation and test. No exact duplicate. Word 8-gram overlap (limit 0.10) is 0.0000 (max) in all six pairs (validation, test, and the 3 benchmarks against train; test against validation). No record is removed. The check is not vacuous: dev_id against train gives a mean of 0.596.
- Super-NaturalInstructions: 47 of the 119 held-out tasks are classification tasks with a fixed label set. 6 of them have the license "Unknown" and are excluded, so 41 remain. Cap: the first 100 instances of each task in file order, as in the official Tk-Instruct loader.
- Metrics (`crux/metrics.py`, `docs/metrics.md`): accuracy, macro F1, ECE (15 bins), counterfactual paired accuracy (by flip, stay, relevant stay, `unit_depth`, and edit type by orig label), paraphrase consistency, option-order invariance (`permute_options`), unseen-rule accuracy (by feature count and depth), and the majority baseline (train labels only). `evaluate` reports each source and `meta.subset` separately when a file mixes them.
- Baselines (accuracy, higher is better; validation): majority binary 0.348, multi-choice 0.298; fact-count probe 0.389 (dev_id 0.454). Per-composition majority values are in `docs/splits.md`.
- Protocol decision: Step 4 reads the test split only to build, hash, and check it. The manifest and the docs hold no label count and no model statistic of test. Step 8 computes the test baselines with the same code. An earlier build of this step scored the fact-count probe and the majority on test. These values are deleted, and nothing used them.
- Correction of Step 3: only 8.5% of the binary stays and 20.4% of the multi-choice stays have `meta.edit_relevant`. `docs/generator_spec.md` section 6 said "80% or more". The generator selects stays by a looser test (any derived value of the queried entity changes). The spec is corrected. The data and the labels do not change. Relevant-stay pairs in validation: 116 binary, 111 multi-choice.
- Contamination (`crux/contamination.py`, `docs/contamination.md`): the test split is scanned against all 66 files of Dolma v1.7 `tulu_flan` (29.1 GB, 11.0B words). The RuleTaker examples of the SNI tasks 966 and 967 occur verbatim (591 of 596 flagged), so Ettin saw RuleTaker text. 1,173 ProofWriter depth-5 records (67 theories) share more than 0.10 of their 8-grams (max 0.20). They share the RuleTaker sentence frames and names, not whole theories. Following the pre-registered rule, the build removes them (`configs/step4_contamination_removed.txt`), and the test split is frozen again: 42,221 records, depth 5 18,857. The synthetic test block and Birds-Electricity share no 8-gram. ParaRules has a max share of 0.011.
- A critical review found a build that overwrote the frozen files before the hash check, a silent loss of unseen-rule accuracy on mixed files, a crash on dev_id, paired slices by `depth`, and tests that did not fail on broken logic. All are fixed. 344 tests pass.
- `pytest.ini` sets `pythonpath = .`, so the plain `pytest` command imports `crux` again.
- Colab notebook: `notebooks/step4_splits.ipynb`.

**Carry-over from earlier steps:**
- Run an n-gram overlap check of the frozen test split against the public Ettin `tulu_flan` shards (Step 1, `docs/contamination.md`).
- Count again the Super-NaturalInstructions tasks with license "Unknown" in the classification subset, and exclude them (Step 0).
- Fill in the `null` split sizes, hashes, and n-gram settings in `configs/protocol.yaml` (Step 0).
- Decide a cap per task for Super-NaturalInstructions: 41 tasks, 142,350 records, and the task sizes are very different (Step 2).
- The converters default to depth 3 for train and dev. Depth 5 stays for the test split only. The Step 1 smoke test trained on RuleTaker depth 5, but that model is not used again. Birds-Electricity and ParaRules need converters (Step 2).
- Synthetic data: every val unit has an unseen template group, vocabulary, and composition at the same time, and depths 4–5 occur only in val and test. Keep a separate in-distribution dev set of train-block units (not called "validation"), so that Step 5 can compare seen and unseen accuracy. Report paired accuracy on relevant stays (`meta.edit_relevant`) separately (Step 3).
- Synthetic binary records have 0.357 Unknown labels in the train block (orig records 0.333). Report the majority-class baseline per block (Step 3).
- Synthetic edit records in the 7 compositions without a True↔False flip (c00, c01, c02, c04, c06, c08, c09) are 0.53 Unknown in train and 0.56 in val (other compositions about 0.34). The composition shows in the rule types. Report the majority baseline per composition too (Step 3).
- Report a fact-count baseline: a classifier on the true, false, and missing facts of the queried entity on the query path scores 0.383 (val) and 0.432 (train halves) on balanced orig records (chance 0.333). Random facts with the same label give 0.47, so this signal comes from the logic, but a model can learn it (Step 3).

## Step 5: Architecture A (Official) and Architecture B: Rule Compiler

- [x] Train the official A on the frozen train split (3 seeds).
- [x] Extend Ettin-17M with an instruction-conditioned rank-8 hypernetwork. The hypernetwork generates low-rank parameters for the decision head.
- [x] Ablation: the hypernetwork also generates low-rank adapters for the top K encoder layers.
- [x] Control **A-param**: give A an unconditioned LoRA with the same number of trainable parameters as B. This control shows if the conditioning helps, or only the extra parameters help.
- [x] Train with the same data and compute budget as A. Compare the accuracy, the generalization, the latency, and the parameter count on the validation split.

**Done (2026-10-18):**
- Sweep: 72 of 72 runs on Colab L4, 0 failed (2026-10-17 09:06 to 2026-10-18 04:41, about 55 compute units). Full output of `python -m crux.aggregate runs/step5`: `runs/step5/aggregate.txt`.
- Selected trials (best mean validation unseen_rule_accuracy over 3 seeds): A t0, B t0, B-abl t0 (lr 3e-5, wd 0.01), A-param t4 (lr 3e-5, wd 0.1). All trials with lr 5e-5 or more stay near chance for most seeds.

Validation, selected trial, mean ± std over 3 seeds (higher is better; majority baseline 0.333, dev_id majority 0.347):

| Arch | unseen_rule_accuracy | counterfactual_paired_accuracy | dev_id accuracy |
|---|---|---|---|
| A | 0.561 ± 0.192 | 0.388 ± 0.224 | 0.598 ± 0.206 |
| B | 0.483 ± 0.087 | 0.285 ± 0.102 | 0.483 ± 0.100 |
| B-abl | 0.603 ± 0.099 | 0.432 ± 0.117 | 0.636 ± 0.155 |
| A-param | 0.618 ± 0.058 | 0.453 ± 0.075 | 0.640 ± 0.083 |

Difference to A, unseen_rule_accuracy, 95% paired bootstrap CI over records (higher is better): B −0.079 [−0.086, −0.072], B-abl +0.041 [+0.035, +0.048], A-param +0.056 [+0.050, +0.063].

Parameters and latency on the L4, batch size 1 (lower is better for latency):

| Arch | params_total | Extra vs A | p50 ms (lower is better) | p95 ms (lower is better) |
|---|---|---|---|---|
| A | 16,863,745 | 0 | 9.66 | 10.28 |
| B | 17,146,945 | 283,200 | 10.82 | 11.53 |
| B-abl | 18,744,385 | 1,880,640 | 11.70 | 12.42 |
| A-param | 17,146,625 | 282,880 | 13.90 | 14.86 |

- Result: the instruction conditioning does not help at this budget. B (head only) is worse than A. B-abl is better than A, but A-param, with the same extra parameters as B and 6.6 times fewer than B-abl, is better than B-abl. The gain comes from the extra encoder parameters, not from the conditioning.
- Limitation: the seed spread is large (A std 0.192: seeds 0.694, 0.341, 0.649). The bootstrap CI covers the record sample only, not the seed variance. All differences to A are smaller than the std of A, so the order of A, B-abl, and A-param is weak evidence. Per seed pair (s0 / s1 / s2), B is lower than A in 2 of 3 pairs (0.536 / 0.530 / 0.382 against 0.694 / 0.341 / 0.649).
- Limitation: 6000 steps is short. Many runs stay at chance for 2,000 to 4,000 steps, and some never leave chance. More steps would probably decrease the seed spread.
- A-param latency includes the unmerged LoRA. A merge of the static LoRA into the weights gives the latency of A.

**Rerun at 12000 steps (2026-10-18), the final Step 5 result:** the results above are from 6000 steps and 6 trials, and their seed spread is too large for a fair comparison. The rerun uses `training_steps` 12000 and one trial (lr 3e-5, wd 0.01) for all 4 architectures: 12 runs on Colab L4, 0 failed, 18 compute units (balance after: 49.8). Steps 6 and 7 use the same budget and the same single trial. The 6000-step runs are in `runs/step5_6000/`. Full output: `runs/step5/aggregate.txt`.

Validation, mean ± std over 3 seeds (higher is better; majority baseline 0.333, dev_id majority 0.347):

| Arch | unseen_rule_accuracy | Seeds s0 / s1 / s2 | counterfactual_paired_accuracy | dev_id accuracy |
|---|---|---|---|---|
| A | 0.661 ± 0.067 | 0.589 / 0.720 / 0.675 | 0.504 ± 0.087 | 0.711 ± 0.079 |
| B | 0.661 ± 0.058 | 0.596 / 0.708 / 0.678 | 0.504 ± 0.082 | 0.708 ± 0.092 |
| B-abl | 0.584 ± 0.192 | 0.703 / 0.687 / 0.363 | 0.413 ± 0.229 | 0.640 ± 0.231 |
| A-param | 0.576 ± 0.189 | 0.673 / 0.359 / 0.697 | 0.404 ± 0.224 | 0.625 ± 0.214 |

Difference to A, unseen_rule_accuracy, 95% paired bootstrap CI over records (higher is better): B −0.001 [−0.007, +0.005], B-abl −0.077 [−0.084, −0.071], A-param −0.085 [−0.091, −0.079].

Latency on the L4, batch size 1, p50 ms (lower is better): A 9.53, B 10.68, B-abl 11.93, A-param 14.37 (A-param with the unmerged LoRA).

- Result: B is equal to A (the CI includes 0) at 12% more latency. The instruction-conditioned head gives no gain.
- B-abl and A-param (extra encoder parameters) stay at chance in 1 of 3 seeds. Their other 2 seeds (B-abl 0.703, 0.687; A-param 0.673, 0.697) are above the mean of A (0.661). So the extra encoder parameters can help, but they make the training less reliable. The conditioning (B-abl) gives no gain over the static control (A-param).
- Compared with 6000 steps: A rises from 0.561 ± 0.192 to 0.661 ± 0.067, and no run of A or B stays at chance. 8 of 12 runs have their best checkpoint before step 12000, so the curves are flat at the end.
- Limitation: 3 seeds and 1 trial. The bootstrap CI covers the record sample only, not the seed variance.
- For Step 6: C builds on B, and B equals A here. The Step 6 controls (A+CF, B+CF) must show if the counterfactual data or the consistency loss gives a gain. Credit: the 49.8 units cover Steps 6 and 7 at about 14 units each (3 configurations x 3 seeds, 12000 steps, 1 trial), with about 20 units in reserve.

**Notes from the run:**
- Files: `crux/models.py` (A, B, B-ablation `hyper_layers: K`, A-param), `crux/train.py` (train, resume, evaluate, latency), `crux/aggregate.py` (selection, mean ± std, paired bootstrap CI), `configs/step5.yaml` (4 architectures, 6 trials), `tests/test_models.py`, `notebooks/step5_train.ipynb`.
- Budget: `budget.training_steps` is 6000 (2.5 epochs), frozen on 2026-10-16. The Step 1 curve (dev 0.66 at step 3000, best 0.856 near step 21,800) shows that 3000 steps probably under-trains A. T4 timing run (200 steps, seconds per step, lower is better): A 0.320, B 0.334, B-abl 0.340, A-param 0.413. No error, no OOM (peak 11.7 GiB), no non-finite loss. Estimate for 72 runs with evaluation: about 50 T4 GPU-hours.
- B-abl fix (commit `293b426`): on the T4, all 3 seeds of B-abl trial 0 stopped with a non-finite fp16 loss (steps 507 to 1025). The generated encoder factors grew to 25 times their init scale, and the Wo delta drove the residual stream to the fp16 limit. The encoder factors are now soft-bounded (`limit * tanh(v / limit)`). A, B, and A-param stay bit-identical to `b7220bf` (test), so their runs stay valid. The 3 failed B-abl runs start again from step 0.
- Search change (2026-10-16, before any run of trials 1 to 5): trial 1 lr 2e-5 → 6e-5, trial 5 lr 2e-5 / weight_decay 0.1 → 7e-5 / 0.01. The same 6 trials for all architectures. Reason: in the first 5 runs of trial 0 (lr 3e-5), validation accuracy stays at chance (0.33) for 1,000 to 4,000 steps, every best checkpoint is at step 6000, and A gives 0.663, 0.607, and 0.373 over the 3 seeds. A lower lr would probably not leave chance in 6000 steps. Step 1 showed that 1e-4 fails, so the maximum is 7e-5. Limitation for the report: 6000 steps is short for these models (the curves still rise at the end).
- Restart with `new_lr_mult` 1 (2026-10-17): the sweep stopped at 23 of 72 runs. Many runs stayed at chance (lr 6e-5 for all 6000 steps). Diagnosis on an L4 (A, trial 0, 3 seeds, 2500 steps, 2000 validation records): mean pooling is not the cause (option and [CLS] pooling stay at chance, mean pooling is the best). With `new_lr_mult` 1, the curves leave chance earlier. T4 check, A trial 0, 6000 steps, validation accuracy s0 / s1 / s2 (higher is better): mult 1 0.545 / 0.637 / 0.558 (mean 0.580), mult 10 0.663 / 0.607 / 0.373 (mean 0.548). Mult 1 has the higher mean and a smaller spread, so all 72 runs start again with mult 1. The 23 old runs are in `runs/step5_mult10/` (not used for results). The check runs are in `runs/mult1/`.
- Sweep: `crux/colab_sweep.py` runs the 72 runs on Colab VMs without supervision (2 VMs: Colab refuses a third). From the restart with `new_lr_mult` 1, all 72 runs use L4 VMs (fp16, as on the T4), because an L4 is faster at a similar cost. The T4 timing above is only for reference. The latency comparison uses the L4 values. Status: `runs/step5/sweep_status.json`. Restart after a reboot: `nohup caffeinate -i .venv/bin/python -m crux.colab_sweep >> runs/step5/driver.out 2>&1 &`.
- `num_workers` is 0: the dataset holds a lambda, so a worker process fails with spawn (macOS) and forkserver (Python 3.14 on Linux).

**Carry-over from earlier steps:**
- Set `budget.training_steps` in `configs/protocol.yaml` before you train A. Use the same value for all configurations (Step 0).
- Do not use lr 1e-4 for Ettin-17M. The dev loss is lowest near epoch 6 and then rises. Include lr values near 3e-5 in the 6-trial search (Step 1).
- Report the majority-class baseline with every accuracy. In ProofWriter OWA depth 3, "Unknown" is 44.5% of the train records (Step 2).
- Training uses 32 records per batch in all modes (`FixedBatchSampler`). Equal steps then give equal data (Step 2).
- Token limits (`configs/step2_data.yaml`): `max_length` 640, `max_instruction_length` 448, `max_state_length` 256. No synthetic record is truncated (Step 3 report: a sample of 2,000 records).
- Load the data only with `crux.splits.load_split` (train, dev_id, validation). Select on validation only. Report dev_id (seen rules) next to validation (unseen rules) (Step 4).
- `crux.metrics.evaluate(records, probs, train=train)` gives all metrics. Slice paired metrics by `unit_depth`, not `depth` (orig and edit differ in depth in 165 of 5,053 validation pairs) (Step 4).

## Step 6: Architecture C: Counterfactual Learning

- [x] Extend B with paired counterfactual training, the supervised decision loss, and a consistency loss for logically equivalent instructions.
- [x] Control **A+CF**: train A on the same counterfactual data. This control separates the effect of the data from the effect of the architecture.
- [x] Ablation **B+CF**: train B on the counterfactual data without the consistency loss. This ablation separates the effect of the data from the effect of the loss.
- [x] Compare C, B, A+CF, and B+CF on the validation split.

**Done (2026-10-19):**
- Definition of "counterfactual data" (CF): the frozen train split already holds both members (orig and edit) of every pair and 1 to 3 paraphrases of each member, and A and B trained on all of them in a random order. The protocol requires the same data for all configurations. Thus CF is the order of the same records: `UnitBatchSampler` (`crux/data/batching.py`) shuffles whole units and cuts batches of 32 records, so the orig and the edit record of each pair are always in the same batch (0 of 37,823 pairs split), and the paraphrases of a unit are in the same batch (a batch cut separates about 7.4% of the paraphrase groups). The records per epoch and the batches per epoch (2,364) are those of A and B.
- C: B + CF + cross-entropy + a consistency loss: the mean Jensen-Shannon divergence of each paraphrase group in the batch to the mean distribution of the group (`consistency_loss` in `crux/train.py`), weight 1.0. The weight is a fixed choice: the budget has 1 trial, so it is not searched. A+CF: A + CF. B+CF: B + CF, without the consistency loss.
- Budget: the same as the Step 5 rerun: 12000 steps, 32 records per batch, 1 trial (lr 3e-5, wd 0.01), 3 seeds, Colab L4, fp16. A and B are the Step 5 runs (`runs/step6/A` and `runs/step6/B` link to `runs/step5`). A review confirmed that the unpaired path stays bit-identical to commit `c4f3795` (tiny A and B runs give byte-identical `val_probs` and `train.log`). VM software: torch 2.11.0, transformers 5.18.0 (the Step 5 versions are not recorded).
- Sweep: 9 of 9 runs on Colab L4, 0 failed (2026-10-19 11:39 to 17:36), commit `9152dd3`. About 14.6 compute units (balance 49.78 before, 35.16 after), including a smoke run and 2 runs that started again after a lost VM. Full output: `runs/step6/aggregate.txt`, `aggregate_ref_B.txt`, `aggregate_ref_B+CF.txt`.

Validation, mean ± std over 3 seeds (higher is better; majority baseline 0.333, dev_id majority 0.347):

| Arch | unseen_rule_accuracy | Seeds s0 / s1 / s2 | counterfactual_paired_accuracy | paraphrase_consistency | dev_id accuracy |
|---|---|---|---|---|---|
| A | 0.661 ± 0.067 | 0.589 / 0.720 / 0.675 | 0.504 ± 0.087 | 0.698 ± 0.089 | 0.711 ± 0.079 |
| B | 0.661 ± 0.058 | 0.596 / 0.708 / 0.678 | 0.504 ± 0.082 | 0.734 ± 0.062 | 0.708 ± 0.092 |
| A+CF | 0.530 ± 0.147 | 0.652 / 0.570 / 0.367 | 0.346 ± 0.176 | 0.615 ± 0.047 | 0.557 ± 0.193 |
| B+CF | 0.475 ± 0.109 | 0.508 / 0.563 / 0.353 | 0.279 ± 0.122 | 0.507 ± 0.078 | 0.492 ± 0.135 |
| C | 0.504 ± 0.074 | 0.541 / 0.419 / 0.552 | 0.311 ± 0.071 | 0.569 ± 0.010 | 0.539 ± 0.117 |

ECE, mean over 3 seeds (lower is better): A 0.070, B 0.082, A+CF 0.058, B+CF 0.093, C 0.063.

Differences, 95% paired bootstrap CI over records (higher is better):

| Contrast | Question | unseen_rule_accuracy | counterfactual_paired_accuracy |
|---|---|---|---|
| C − A | the protocol test | −0.158 [−0.165, −0.151] | −0.193 [−0.202, −0.184] |
| C − B | C against its base | −0.157 [−0.164, −0.150] | −0.193 [−0.202, −0.184] |
| A+CF − A | the effect of CF on A | −0.132 [−0.139, −0.125] | −0.158 [−0.167, −0.149] |
| B+CF − B | the effect of CF on B | −0.186 [−0.193, −0.178] | −0.225 [−0.234, −0.215] |
| C − B+CF | the effect of the consistency loss | +0.029 [+0.023, +0.036] | +0.032 [+0.024, +0.039] |

Parameters (equal to A or B: CF adds no parameter) and latency on the L4, batch size 1, p50 ms (lower is better): A 9.53, A+CF 9.81, B 10.68, B+CF 10.92, C 10.78.

- Result: the paired batches decrease the accuracy of A and of B. C is not better than A or B. Its CIs exclude 0 in the negative direction for both primary metrics. Thus Step 6 gives no reliable improvement over A. It gives a reliable decrease.
- The consistency loss gives a small gain over B+CF (+0.029, CI excludes 0), but C is better than B+CF in only 2 of 3 seed pairs (0.541 / 0.419 / 0.552 against 0.508 / 0.563 / 0.353). The seed spread (B+CF std 0.109) is larger than the gain, so this is weak evidence. Paraphrase consistency: C 0.569 against B+CF 0.507, but both are lower than B (0.734), because the consistency metric also depends on the accuracy.
- Cause (probable): the paired runs learn more slowly. They leave chance later than A and B, and 8 of 9 paired runs have their best checkpoint at step 11000 or 12000 (A and B: 2 of 6 at step 12000), so the curves still rise at the end. A paired batch of 32 records holds about 11 units (3.0 records per unit on average), against 32 nearly independent records for A and B. The consistency term is small (the mean over 100 steps is at most 0.019, against a cross-entropy of 0.8 to 1.2), so the loss does not cause the slowdown: A+CF has no consistency loss and decreases too.
- Limitation: 3 seeds, 1 trial, and a fixed consistency weight. The bootstrap CI covers the record sample only, not the seed variance. The result applies to this budget (12000 steps). A longer run, or a mix of paired and random batches, could give another result. These would need the same budget for all configurations.
- For Step 7: D extends C. C is worse than B at this budget, so compare D also with B and with B + the D parts (no CF), if the credit permits (35.2 units left, about 14 for Step 7).

**Notes from the run:**
- Files: `crux/data/batching.py` (`UnitBatchSampler`, `pair_units`, `make_dataloader(..., paired=)`), `crux/train.py` (`consistency_loss`, `paraphrase_rows`, the `train.paired` and `train.consistency_weight` keys; an architecture entry can set train keys), `crux/aggregate.py` (`--ref` for contrasts with another reference, and the `paraphrase_consistency` column), `configs/step6.yaml`, `notebooks/step6_train.ipynb`. 7 new tests. 377 tests pass.
- Smoke run (L4, 60 steps of C, a path with "+"): fp16 path, finite loss, `consistency` in `train.log`, and the download of a path with "+" work.
- Sweep driver: `crux6-s0` was lost two times (12:02 and 12:36), each time when a setup exec on `crux6-s1` hung for 20 to 30 minutes. The driver is one thread, so it did not probe the other VM in that time. The setup and verify calls now stop after 300 s (commit `0c96a3a`), and the driver ran with `--tick 120 --stall 600`. The lost runs started again from step 0. Do not run a training smoke test on the Mac (MPS): it froze the Mac.

## Step 7: Architecture D: CRUX

- [x] Extend C with independently encoded, cacheable `state` and `instruction` representations, and a permutation-equivariant candidate-scoring module.
- [x] Implement shared computation, so that one cached `state` serves many questions.
- [x] Ablation **C+cache**: caching only. Independent encoding can decrease accuracy compared with joint encoding. Measure this trade-off.
- [x] Ablation **C+equivariant**: candidate scoring only.
- [x] Evaluate unseen-rule generalization, option-order invariance, memory consumption, and inference latency (with and without a cache hit).

**Done (2026-10-22):**
- D = C + the split encoding + the set-attention block (`crux/models.py`):
  - Split encoding (`joint_layers` J = 4): the bottom 3 of the 7 Ettin layers encode the state, the instruction, and each option as separate sequences ([CLS] text [SEP], positions from 0). These bottom-layer states are the cache: they do not depend on the question options or on each other. The top 4 layers (3 to 6, with the 2 global layers 3 and 6) read [state ; instruction ; option] for each option. The real tokens move to the front of the concatenation, so the RoPE positions and the sliding window do not depend on the padding. J is a fixed choice (1 trial). The split encoding adds no parameters.
  - Set-attention block (`option_attention`): one pre-norm multi-head self-attention block over the pooled vectors of the options of a record, with no positional information and with a key mask for the padded options. The output projection starts at zero, so the block starts as the identity. It adds 263,680 parameters. The scores are permutation-equivariant by construction.
  - The hypernetwork input z has the same definition as in C (the mean final hidden state over the instruction tokens). Thus C+cache differs from C only in the encoding.
- Configurations (`configs/step7.yaml`): D (C + both parts), C+cache (C + the split encoding), C+equivariant (C + the block, joint encoding), and the extra control D-CF (B + both parts, without the paired batches and without the consistency loss). D-CF exists because C is worse than B at this budget (Step 6). A and B are the Step 5 runs, and C is the Step 6 run (`runs/step7/A`, `B`, `C` are links).
- Budget: the same as Steps 5 and 6: 12000 steps, 32 records per batch, 1 trial (lr 3e-5, wd 0.01), 3 seeds, Colab L4, fp16. Separate mode truncates 0 records of train and validation (the longest state has 242 tokens with the limit 256, the longest instruction has 424 with the limit 448).
- Sweep: 12 of 12 runs on Colab L4, 0 failed (2026-10-21 21:55 to 2026-10-22 02:30), commit `c3acf20`. 13.0 compute units, including an L4 smoke run of D (balance 35.16 before, 22.16 after). VM software: torch 2.11.0, transformers 5.20.1 (Step 6: 5.18.0). Full output: `runs/step7/aggregate.txt` and `aggregate_ref_{B,C,D-CF,C+cache,C+equivariant}.txt`.

Validation, mean ± std over 3 seeds (higher is better; majority baseline 0.333, dev_id majority 0.347):

| Arch | unseen_rule_accuracy | Seeds s0 / s1 / s2 | counterfactual_paired_accuracy | paraphrase_consistency | dev_id accuracy |
|---|---|---|---|---|---|
| A | 0.661 ± 0.067 | 0.589 / 0.720 / 0.675 | 0.504 ± 0.087 | 0.698 ± 0.089 | 0.711 ± 0.079 |
| B | 0.661 ± 0.058 | 0.596 / 0.708 / 0.678 | 0.504 ± 0.082 | 0.734 ± 0.062 | 0.708 ± 0.092 |
| C | 0.504 ± 0.074 | 0.541 / 0.419 / 0.552 | 0.311 ± 0.071 | 0.569 ± 0.010 | 0.539 ± 0.117 |
| C+cache | 0.446 ± 0.090 | 0.510 / 0.484 / 0.343 | 0.246 ± 0.092 | 0.718 ± 0.084 | 0.432 ± 0.074 |
| C+equivariant | 0.564 ± 0.114 | 0.597 / 0.659 / 0.438 | 0.388 ± 0.125 | 0.636 ± 0.098 | 0.608 ± 0.169 |
| D | 0.451 ± 0.075 | 0.367 / 0.476 / 0.511 | 0.254 ± 0.081 | 0.645 ± 0.039 | 0.439 ± 0.080 |
| D-CF | 0.398 ± 0.045 | 0.386 / 0.447 / 0.360 | 0.194 ± 0.052 | 0.580 ± 0.152 | 0.376 ± 0.042 |

ECE, mean over 3 seeds (lower is better): A 0.070, B 0.082, C 0.063, C+cache 0.024, C+equivariant 0.071, D 0.040, D-CF 0.081.

Differences, 95% paired bootstrap CI over records (higher is better). "Seed pairs" counts the seed pairs (s0, s1, s2) in which the first model is higher:

| Contrast | Question | unseen_rule_accuracy | counterfactual_paired_accuracy | Seed pairs |
|---|---|---|---|---|
| D − A | the protocol test | −0.210 [−0.218, −0.203] | −0.250 [−0.260, −0.240] | 0 of 3 |
| D − C | D against its base | −0.052 [−0.060, −0.045] | −0.057 [−0.065, −0.049] | 1 of 3 |
| C+cache − C | the cost of the split encoding | −0.058 [−0.065, −0.051] | −0.065 [−0.073, −0.057] | 1 of 3 |
| D − C+equivariant | the cost of the split encoding, with the block | −0.113 [−0.120, −0.106] | −0.134 [−0.143, −0.126] | 1 of 3 |
| C+equivariant − C | the effect of the block | +0.061 [+0.054, +0.067] | +0.077 [+0.070, +0.085] | 2 of 3 |
| D − C+cache | the effect of the block, with the split encoding | +0.006 [−0.001, +0.012] | +0.008 [+0.000, +0.015] | 1 of 3 |
| D − D-CF | the effect of CF and the consistency loss, in D | +0.054 [+0.047, +0.061] | +0.060 [+0.053, +0.068] | 2 of 3 |
| D-CF − B | the effect of both D parts, without CF | −0.263 [−0.271, −0.255] | −0.310 [−0.321, −0.299] | 0 of 3 |

Option-order invariance (validation, a fixed random permutation of the options of each record; 8,407 to 8,422 of the 10,106 records change their order, depending on the seed): the argmax agreement is 1.000 for all 21 runs. The mean absolute probability difference per run is at most 1.1e-6 (the Step 7 runs on the L4 in fp16; A, B, and C with `--eval-only` on the Mac CPU in fp32). The shuffled-order unseen_rule_accuracy differs from the canonical-order value by at most 0.0011. A, B, and C score each option alone, so they are order-invariant by construction too. Thus this metric does not separate D from the other models. The block adds an interaction between the options and keeps the invariance.

Memory and latency on the L4, mean over 3 seeds (lower is better in all columns):

| Arch | Model MB fp16 | Peak CUDA MB above the model, bs1 | bs1 p50 ms, no cache | bs1 p50 ms, state cached | bs1 p50 ms, state and instruction cached |
|---|---|---|---|---|---|
| A | 33.73 | n/a | 9.53 | n/a | n/a |
| C | 34.29 | n/a | 10.78 | n/a | n/a |
| C+cache | 34.29 | 39.8 | 19.98 | 15.46 | 11.51 |
| C+equivariant | 34.82 | 39.3 | 11.12 | n/a | n/a |
| D | 34.82 | 39.8 | 19.80 | 15.67 | 11.76 |
| D-CF | 34.82 | 39.5 | 20.25 | 16.04 | 12.06 |

Cache size, fp16, mean over the distinct validation items (lower is better): 37.3 KB per state (72.8 tokens), 100.1 KB per instruction (195.5 tokens). Validation has 10,106 records, 5,642 distinct states (1.79 records per state), and 6,728 distinct instructions.

- Result: D is not better than A. Its CIs exclude 0 in the negative direction for both primary metrics (−0.210 and −0.250), and D is lower than A in all 3 seed pairs. Thus Step 7 gives no reliable improvement over A. D is also lower than its base C (−0.052).
- The split encoding decreases the accuracy: −0.058 from C to C+cache, and −0.113 from C+equivariant to D. Without CF, the loss is larger: D-CF is 0.263 lower than B. C+cache s2 and D s0 stay near chance (0.343, 0.367).
- The set-attention block helps with the joint encoding (C+equivariant − C = +0.061, 2 of 3 seed pairs), but not with the split encoding (D − C+cache = +0.006, CI includes 0). C+equivariant is the best Step 7 model, but it is 0.097 lower than A. The gain over C is smaller than the seed spread of C+equivariant (std 0.114), so it is weak evidence.
- CF reverses its sign with the split encoding: in Step 6, CF decreased B (B+CF − B = −0.186). Here, D (with CF and the consistency loss) is 0.054 higher than D-CF. CF does not make D better than A or B.
- The cache does not make the latency at batch size 1 better than C. Without a cache, D needs 19.80 ms (C: 10.78 ms), because the bottom layers run on 3 sequences of each record and the batch-size-1 time on the L4 is mostly kernel launch time. With the state and the instruction cached, D needs 11.76 ms, which is still 9% more than C. Only 3 of 7 layers are cached: the top 4 layers run on the full concatenation for each option. A larger cache share (smaller J) would be faster, but the split encoding already costs accuracy at J = 4.
- The peak CUDA memory at batch size 1 is the same for the joint and the split encoding (39.3 to 39.8 MB above the model). The peak memory of A, B, and C is n/a: their `--eval-only` check ran on the CPU.
- Cause (probable) of the accuracy loss of the split encoding: the bottom layers see the state, the instruction, and the options without each other, the top layers see positions and 3 [CLS]/[SEP] pairs that the pretraining did not have, and the runs learn slowly. 6 of 6 D and C+equivariant runs and 2 of 3 C+cache runs have their best checkpoint at step 11000 or 12000, so the curves still rise at the end.
- Limitation: 3 seeds, 1 trial, a fixed J = 4, and a fixed consistency weight. The bootstrap CI covers the record sample only, not the seed variance. The result applies to this budget (12000 steps).
- For Step 8: benchmark D-CF too (the list has A, A-param, A+CF, B, B+CF, C, D, and the two ablations of D). Credit: 22.16 units left for Steps 8 and 9, including the external baselines (Ettin-32M and DeBERTa-v3-xsmall, 3 seeds each).

**Notes from the run:**
- Files: `crux/models.py` (`joint_layers`, `option_attention`, `encode_bottom`, `forward(..., state_cache=, instruction_cache=)`), `crux/train.py` (`input_mode`: separate mode when J > 0; `order_invariance`, `memory`, and `latency.cache_hit` in `metrics.json`; `--eval-only <run_dir>` writes `order_invariance.json` for old runs), `crux/aggregate.py` (a table for the order invariance, the memory, and the cache), `crux/colab_sweep.py` (downloads `resolved_config.yaml` too), `configs/step7.yaml`, `notebooks/step7_train.ipynb`. 15 new tests. 392 tests pass. A, B, and C stay bit-identical (tiny CPU runs before and after give identical `val_probs`). A review agent checked the padding, the option mapping in batches with 2 to 8 options, the cache equality (difference 0), and the permutation equivariance on the real Ettin-17M.
- The cache size assumes fp16 storage. Under autocast the hidden states are fp32, so a cache of these states without a cast is 2 times larger.
- The separate-mode evaluation budget (`row_size` = Ls + Li + K × Lo) does not count the top-layer tokens (K × (Ls + Li + Lo)). Thus the evaluation batches of the split models are larger: peak 631 MB against 171 MB for C+equivariant, and their per-record batch latency (1.46 ms against 1.68 ms) is not comparable. Use the batch-size-1 values.
- The per-type paired accuracies are in each `metrics.json` (`paired_accuracy/edit_type=...`). The Step 3 count of the multi-choice rule-edit pairs per block (validation: c23 only) is not in this report. Do it in Step 8 for all models.
- Sweep driver: one setup exec on `crux7-s1` stopped after 420 s, and the next setup worked (2 minutes lost). No VM was lost.

**Carry-over from earlier steps:**
- Binary options use one canonical order in all sources (positive label first, then negative, then "Unknown"). Shuffle the option order when you measure order invariance (Step 2).
- Multi-choice pairs with a rule edit occur only in compositions with `pri`. Validation has only c23 (31 units), so report the validation results for these pairs as one-composition results. Count these pairs per block in the report (Step 3).

## Step 8: Benchmark All Models

- [x] Evaluate A, A-param, A+CF, B, B+CF, C, D, the two ablations of D, and the external baselines on the frozen test split. Evaluate also on RuleTaker, ProofWriter, BoardgameQA, DynaBench, and the Super-NaturalInstructions held-out classification tasks (only the benchmarks that pass the Step 0 license check). Deviation: the external baselines are not evaluated (scope set by the user; see below).
- [x] Use one evaluation script and one protocol for all models.
- [x] Generate comparison tables and plots for accuracy, paired counterfactual accuracy, calibration, generalization, and latency. Write the better direction in each table title, for example "(higher is better)".
- [x] Add 95% CIs and the paired bootstrap test for each comparison with A.
- [x] Apply the criteria from Step 0. Report all results, including the comparisons where the CI includes 0.

**Carry-over from earlier steps:**
- Limitations: A, B, C, and D share the Ettin initialization, so a contamination affects them equally. DeBERTa-v3-xsmall has different pretraining data (Step 1).
- DynaBench has only 543 test examples, so its CIs are wide (Step 0).
- Long inputs: DynaBench has up to 5,486 joint tokens. With the Step 2 limits (`max_length` 640, `max_instruction_length` 448, `max_state_length` 256), joint mode truncates 350 of 543 DynaBench records (rules 314, state 136), and separate mode truncates 397. BoardgameQA: joint mode with options truncates 772 of 37,500 records, and separate mode truncates 1,832 (rules 1,807, state 45). The question stays whole. Set larger evaluation limits (Ettin supports about 8K tokens). Use the same limits for all models (Step 2).
- Shuffle the option order when you measure order invariance (Step 2).
- Report each edit type split by the orig label. The mix of edit types depends on the answer: for binary units, `condition_negate` is 0.65 of the units that start at Unknown and 0.25 of those that start at True or False, and `rule_remove` is 0.001 and 0.14. `fact_remove` from Unknown has the flip share 0.29 (a depth-0 Unknown is always a stay) (Step 3).
- Do not report per-type results of `exception_remove` and `priority_swap` for val (22 and 4 units) (Step 3).
- Read the test split with `load_split("test", allow_test=True)`, one time. Compute the test baselines (majority with `ref=train`, fact-count probe) at the same time. Report each test subset separately: synthetic, `depth5`, `birds_electricity`, `pararules`, each with its majority baseline (Step 4).
- `depth5` holds 10 questions of depth 6 or 7. `meta.depth` records them (Step 4).
- The sentence frames and names of `depth5` occur in the Ettin pretraining data (the max 8-gram share of a kept record is 0.10). State this for `depth5` and for all Ettin-based models. The Super-NaturalInstructions benchmark is not scanned: report it as "possibly seen in pretraining", or scan it first with `crux.contamination` (Step 4).
- The held-out sets are the open-world versions (True / False / Unknown) from the ProofWriter release, because the synthetic training data has 3 classes. Compare them only with published open-world results (ProofWriter, Tafjord et al. 2021). Do not compare them with the closed-world RuleTaker results (Clark et al. 2020). The test split has no closed-world RuleTaker set (Step 4).

**Done (2026-10-22, minimum scope):**
- Scope (the user's choice, "Minimum, ~3 h"): evaluate the 33 existing runs (11 models × 3 seeds) on the test split and on the 3 benchmarks. Deviation from the protocol: the external baselines (Ettin-32M and DeBERTa-v3-xsmall, 3 seeds each) are not trained and not evaluated. Their code and configurations exist (`configs/step8_baselines.yaml`, `configs/step8_deberta_data.yaml`), so they can run later. Jev: not evaluated (no API access). RuleTaker: the test split has no closed-world RuleTaker set (Step 4), so the ProofWriter open-world sets stand in for it.
- One script for all models: `crux/benchmark.py`. `Data.get` is the only reader of the test split, and it needs `--final`. The test split was read one time, in this run. The evaluation limits are the same for all models (`configs/step8_eval_data.yaml`: `max_length` 5600, `max_state_length` 2400, `max_instruction_length` 5500). 0 records are truncated in all sets.
- Run: 2 Colab L4 VMs, 2026-10-22 09:11 to 10:26, commit `b1948bb`. 33 of 33 runs and the baselines are complete, and no VM was lost. 3.8 compute units (22.16 before, 18.37 after). Each run needs about 2.5 to 3 minutes (test: 42,221 records in 86 to 105 s). The first launch failed before any evaluation (no upload worked, see `b1948bb`), so the test split was not read in that launch.
- Check: the validation values of this script agree with the Step 7 record (D 0.4514, D − A = −0.2100 [−0.2176, −0.2026]).
- Output (not in git, `runs/` is ignored): `runs/step8/benchmark.txt` and `benchmark.json` (all tables, all CIs, the per-subset and per-task values), `runs/step8/eval/<model>/s<k>/bench.json` and `probs_<set>.npz`, `runs/step8/eval/baselines.json`, and 5 plots in `runs/step8/plots` (accuracy, ECE, generalization, paired accuracy, latency against accuracy).

Accuracy, mean ± std over 3 seeds (higher is better). ▲ = the 95% paired bootstrap CI of the difference to A is above 0, ▼ = below 0, no mark = the CI includes 0:

| Model | test synthetic | test depth5 | test birds_electricity | test pararules | BoardgameQA | DynaBench | Super-NaturalInstructions |
|---|---|---|---|---|---|---|---|
| n | 10,086 | 18,857 | 5,270 | 8,008 | 15,000 | 543 | 4,100 |
| majority baseline | 0.336 | 0.429 | 0.748 | 0.500 | 0.367 | 0.508 | 0.442 |
| A | 0.640 ± 0.049 | 0.406 ± 0.091 | 0.552 ± 0.134 | 0.509 ± 0.004 | 0.373 ± 0.005 | 0.499 ± 0.030 | 0.440 ± 0.007 |
| A-param | 0.552 ± 0.156 ▼ | 0.443 ± 0.058 ▲ | 0.705 ± 0.056 ▲ | 0.489 ± 0.026 ▼ | 0.374 ± 0.008 | 0.495 ± 0.092 | 0.435 ± 0.002 |
| A+CF | 0.522 ± 0.142 ▼ | 0.391 ± 0.039 ▼ | 0.548 ± 0.303 | 0.509 ± 0.042 | 0.362 ± 0.012 ▼ | 0.546 ± 0.067 ▲ | 0.436 ± 0.003 |
| B | 0.636 ± 0.056 | 0.452 ± 0.005 ▲ | 0.666 ± 0.225 ▲ | 0.503 ± 0.003 ▼ | 0.367 ± 0.000 ▼ | 0.475 ± 0.014 | 0.434 ± 0.000 |
| B-abl | 0.573 ± 0.177 ▼ | 0.384 ± 0.065 ▼ | 0.673 ± 0.078 ▲ | 0.503 ± 0.022 ▼ | 0.366 ± 0.004 ▼ | 0.529 ± 0.032 ▲ | 0.437 ± 0.002 |
| B+CF | 0.485 ± 0.104 ▼ | 0.402 ± 0.090 | 0.763 ± 0.016 ▲ | 0.499 ± 0.012 ▼ | 0.367 ± 0.005 ▼ | 0.467 ± 0.024 ▼ | 0.442 ± 0.004 |
| C | 0.513 ± 0.068 ▼ | 0.447 ± 0.033 ▲ | 0.669 ± 0.191 ▲ | 0.508 ± 0.014 | 0.368 ± 0.001 ▼ | 0.534 ± 0.049 ▲ | 0.441 ± 0.005 |
| C+cache | 0.433 ± 0.079 ▼ | 0.421 ± 0.024 ▲ | 0.587 ± 0.301 ▲ | 0.455 ± 0.102 ▼ | 0.370 ± 0.004 | 0.514 ± 0.011 | 0.442 ± 0.003 |
| C+equivariant | 0.559 ± 0.092 ▼ | 0.449 ± 0.033 ▲ | 0.681 ± 0.040 ▲ | 0.511 ± 0.012 | 0.368 ± 0.002 ▼ | 0.496 ± 0.022 | 0.441 ± 0.003 |
| D | 0.442 ± 0.062 ▼ | 0.368 ± 0.086 ▼ | 0.771 ± 0.024 ▲ | 0.498 ± 0.033 ▼ | 0.368 ± 0.004 ▼ | 0.510 ± 0.024 | 0.433 ± 0.006 |
| D-CF | 0.404 ± 0.051 ▼ | 0.343 ± 0.040 ▼ | 0.410 ± 0.228 ▼ | 0.494 ± 0.011 ▼ | 0.369 ± 0.004 | 0.479 ± 0.055 | 0.443 ± 0.002 |

The test majority baselines use the train label as the reference. The benchmark majority baselines use the labels of each benchmark. The fact-count probe (trained on the orig binary records of train, higher is better): test 0.380 (n 1,707), validation 0.387, dev_id 0.460.

Test synthetic, the counterfactual metrics, mean over 3 seeds (higher is better):

| Model | counterfactual_paired_accuracy | paraphrase_consistency |
|---|---|---|
| A | 0.481 ± 0.062 | 0.758 ± 0.042 |
| A-param | 0.374 ± 0.191 ▼ | 0.702 ± 0.064 |
| A+CF | 0.339 ± 0.171 ▼ | 0.721 ± 0.050 |
| B | 0.472 ± 0.076 ▼ | 0.743 ± 0.056 |
| B-abl | 0.397 ± 0.214 ▼ | 0.694 ± 0.140 |
| B+CF | 0.290 ± 0.123 ▼ | 0.679 ± 0.087 |
| C | 0.327 ± 0.071 ▼ | 0.679 ± 0.048 |
| C+cache | 0.229 ± 0.080 ▼ | 0.762 ± 0.044 |
| C+equivariant | 0.386 ± 0.109 ▼ | 0.692 ± 0.016 |
| D | 0.241 ± 0.073 ▼ | 0.667 ± 0.050 |
| D-CF | 0.199 ± 0.057 ▼ | 0.700 ± 0.033 |

The bootstrap is for the accuracy metrics only, so the paraphrase_consistency column has no marks. Differences to A on test synthetic (higher is better): D − A = −0.198 [−0.206, −0.191] for unseen_rule_accuracy and −0.240 [−0.250, −0.230] for counterfactual_paired_accuracy. B − A = −0.004 [−0.010, +0.001] and −0.009 [−0.016, −0.001]. C+equivariant − A = −0.081 [−0.088, −0.075] and −0.095 [−0.103, −0.087].

ECE on test synthetic, mean over 3 seeds (lower is better): A 0.075, A-param 0.102, A+CF 0.052, B 0.098, B-abl 0.107, B+CF 0.067, C 0.067, C+cache 0.036, C+equivariant 0.081, D 0.065, D-CF 0.074. The ECE of the other sets is in `benchmark.txt`.

Latency on the L4 at batch size 1, p50 from the training runs (lower is better): A 9.53 ms, A+CF 9.81, B 10.68, C 10.78, B+CF 10.92, C+equivariant 11.12, B-abl 11.93, A-param 14.37, D 19.80, C+cache 19.98, D-CF 20.25 (D with the state and the instruction cached: 11.76, Step 7). Parameters (total = trainable): A and A+CF 16,863,745; B, B+CF, C, and C+cache 17,146,945; A-param 17,146,625; C+equivariant, D, and D-CF 17,410,625; B-abl 18,744,385.

Test synthetic, counterfactual_paired_accuracy per edit type and orig label, mean over 3 seeds (higher is better). Only the cells with n ≥ 40 units are shown. `exception_remove` (69 units) and `priority_swap` (53 units) are too small to report per type:

| Edit type / orig | n | A | B | C | C+equivariant | D | D-CF |
|---|---|---|---|---|---|---|---|
| fact_flip / True | 211 | 0.796 | 0.776 | 0.675 | 0.719 | 0.425 | 0.359 |
| fact_flip / False | 218 | 0.742 | 0.708 | 0.602 | 0.680 | 0.430 | 0.381 |
| fact_flip / Unknown | 69 | 0.787 | 0.739 | 0.614 | 0.652 | 0.517 | 0.420 |
| fact_flip / multi_choice | 862 | 0.616 | 0.610 | 0.376 | 0.456 | 0.225 | 0.166 |
| fact_remove / True | 345 | 0.705 | 0.719 | 0.559 | 0.629 | 0.397 | 0.299 |
| fact_remove / False | 375 | 0.730 | 0.689 | 0.591 | 0.645 | 0.404 | 0.327 |
| fact_remove / Unknown | 107 | 0.604 | 0.567 | 0.539 | 0.564 | 0.505 | 0.402 |
| fact_remove / multi_choice | 193 | 0.389 | 0.361 | 0.197 | 0.278 | 0.140 | 0.114 |
| condition_negate / True | 345 | 0.351 | 0.338 | 0.135 | 0.200 | 0.097 | 0.100 |
| condition_negate / False | 312 | 0.366 | 0.347 | 0.151 | 0.262 | 0.104 | 0.136 |
| condition_negate / Unknown | 636 | 0.525 | 0.529 | 0.316 | 0.398 | 0.279 | 0.172 |
| condition_negate / multi_choice | 166 | 0.231 | 0.331 | 0.183 | 0.235 | 0.145 | 0.153 |
| rule_remove / True | 98 | 0.361 | 0.344 | 0.143 | 0.197 | 0.088 | 0.156 |
| rule_remove / False | 106 | 0.292 | 0.343 | 0.164 | 0.252 | 0.091 | 0.154 |
| rule_remove / multi_choice | 48 | 0.299 | 0.285 | 0.229 | 0.222 | 0.139 | 0.215 |
| threshold_shift / True | 157 | 0.176 | 0.146 | 0.146 | 0.119 | 0.138 | 0.149 |
| threshold_shift / False | 163 | 0.200 | 0.149 | 0.117 | 0.149 | 0.119 | 0.170 |
| threshold_shift / Unknown | 98 | 0.211 | 0.241 | 0.235 | 0.211 | 0.259 | 0.156 |
| fact_number / multi_choice | 236 | 0.168 | 0.134 | 0.113 | 0.129 | 0.123 | 0.114 |
| fact_number / True, False, Unknown | 46 to 54 | 0.135 to 0.203 | 0.111 to 0.206 | 0.058 to 0.191 | 0.086 to 0.156 | 0.130 to 0.177 | 0.057 to 0.136 |

Test depth5, accuracy per question depth, mean over 3 seeds (higher is better). The per-depth majority baseline is not in the output:

| Depth | n | A | A-param | B | C | C+equivariant | D | D-CF |
|---|---|---|---|---|---|---|---|---|
| 0 | 5,774 | 0.355 | 0.364 | 0.402 | 0.360 | 0.388 | 0.404 | 0.264 |
| 1 | 4,247 | 0.343 | 0.389 | 0.381 | 0.376 | 0.391 | 0.351 | 0.278 |
| 2 | 2,730 | 0.431 | 0.464 | 0.478 | 0.480 | 0.479 | 0.355 | 0.376 |
| 3 | 2,249 | 0.467 | 0.535 | 0.535 | 0.544 | 0.528 | 0.352 | 0.426 |
| 4 | 1,990 | 0.495 | 0.556 | 0.544 | 0.574 | 0.543 | 0.346 | 0.458 |
| 5 | 1,857 | 0.507 | 0.550 | 0.536 | 0.579 | 0.536 | 0.354 | 0.470 |
| 6 and 7 | 10 | too few to report | | | | | | |

Results against the Step 0 criteria ("reliable" = the CI of the difference to A excludes 0):
- The primary result: on test synthetic (unseen rules), no model is reliably better than A. B is equal to A (−0.004, CI includes 0). All other models are reliably worse. D is 0.198 lower than A. This repeats the validation result of Steps 5 to 7.
- On counterfactual_paired_accuracy (test synthetic), all models are reliably lower than A. B is the nearest (−0.009).
- ProofWriter open-world sets: the reliable gains over A on depth5 and birds_electricity come with a large seed spread (for example, birds_electricity: A 0.552 ± 0.134, B 0.666 ± 0.225, C+cache 0.587 ± 0.301). The bootstrap CI covers the records, not the seeds, so these gains are weak evidence. Only B+CF (0.763) and D (0.771) are above the birds_electricity majority (0.748), by at most 0.023. On depth5, B, C, C+equivariant, and A-param are 0.014 to 0.023 above the majority (0.429), and the other 7 models are below it. On pararules, all models are at the majority (0.455 to 0.511 against 0.500).
- These ProofWriter values are far below the published open-world ProofWriter results (Tafjord et al. 2021). The models do not transfer from the synthetic generator to the ProofWriter text. Compare them only with open-world results.
- External benchmarks: no model is materially above the majority baseline. BoardgameQA: all models are 0.362 to 0.374 against 0.367, and all 15 subsets are at their majority. Super-NaturalInstructions: 0.433 to 0.443 against 0.442, and no difference to A is reliable. DynaBench: 0.467 to 0.546 against 0.508. A+CF, C, and B-abl are reliably above A (+0.047, +0.036, +0.030), but A is below the majority, the seed spread is large (A+CF ± 0.067), and n = 543. Thus these are not evidence of a transfer.
- Option-order invariance on test (a fixed random permutation): the mean argmax agreement is 0.993 to 1.000 for each model and subset. The lowest value is C+cache on birds_electricity (0.993 ± 0.012). On validation, the agreement is 1.000 for all runs (Step 7). The cause of the small test difference (fp16 on the L4, or near-tied scores on longer inputs) is not examined.
- Calibration: no model is best on all sets. On test synthetic, C+cache has the lowest ECE (0.036), and A has 0.075.
- Latency: A is the fastest model and the most accurate on test synthetic. The D family needs about 2 times the latency of A without a cache.

**Limitations:**
- No external baselines (Ettin-32M, DeBERTa-v3-xsmall), because of the scope. Thus there is no comparison with a model that has different pretraining. Jev is not evaluated (no API access).
- All 11 models share the Ettin-17M initialization, so a contamination affects them equally. The sentence frames and names of `depth5` occur in the Ettin pretraining data (max 8-gram share 0.10). Super-NaturalInstructions is not scanned: treat it as "possibly seen in pretraining".
- DynaBench has 543 records, so its CIs are wide.
- 3 seeds, 1 trial, and 12000 steps for every model. The bootstrap CI covers the record sample only, not the seed variance. On the ProofWriter sets the seed spread is larger than most differences.
- The ProofWriter sets are open-world (True / False / Unknown). Compare them only with published open-world results, not with the closed-world RuleTaker results (Clark et al. 2020).
- The per-depth majority baseline of depth5 is not computed. Thus the rise of the accuracy from depth 0 to depth 5 has no reference. Depths 6 and 7 have 10 questions.
- The count of the multi-choice rule-edit pairs per composition (Step 3 carry-over) is not in the output. The test synthetic multi-choice rule-edit pairs are: `condition_negate` 166, `rule_remove` 48, `threshold_shift` 26, `priority_swap` 2, `exception_remove` 0. The split per composition (test: c18, c28, c31) is still open.

**Notes from the run:**
- Files: `crux/benchmark.py`, `crux/colab_eval.py`, `crux/plots.py`, `configs/step8_eval.yaml`, `configs/step8_eval_data.yaml`, the `--bench` report in `crux/aggregate.py`, and the baseline code (`configs/step8_baselines.yaml`, a non-ModernBERT path for architecture A in `crux/models.py`). 399 tests pass.
- Two driver bugs stopped the first launch before any evaluation: `colab upload` does not make the parent folders, and the start command had wrong quotes (fixed in `b1948bb`).
- `python -m crux.aggregate --bench` failed at the end of the run, because the `__main__` block came before the definition of `bench_report`. The block is now at the end of the file. The report and the plots were then made on the Mac from the downloaded outputs.
- The validation values on the CPU (fp32) differ from the L4 (fp16) by at most 0.0016 (a sample of 1,906 records: A 0, C 3 flips, D 0.001).

## Step 9: Optimize for Mobile

- [ ] Select the architecture with the best validation result, not the best test result.
- [ ] Export it to ONNX. If the hypernetwork generates weights at runtime, check that the export keeps this computation, or pre-compile the parameters for each fixed instruction.
- [ ] Apply INT8 quantization. Compare the INT8 and FP32 versions for accuracy loss.
- [ ] Measure model size, peak RAM, and latency on Colab.
- [ ] Build an Android benchmark with ONNX Runtime Mobile. Measure on the device named in Step 0, and record the device, the OS version, and the thread count.

**Carry-over from earlier steps:**
- Measure on the Google Pixel 8 only. Keep the same Android OS version for all measurements. Record the OS version and the thread count (Step 0).

**Skipped (2026-10-22, the user's decision):**
- Step 9 is not done. Deviation from the protocol: Step 0 pre-registered the mobile measurements (model size, peak RAM, p50/p95 latency, and the INT8 accuracy loss on the Google Pixel 8). They are not measured.
- Reason: no architecture is better than A (Step 8). The best validation result is A (0.661, equal to B), which is the plain Ettin-17M baseline. A mobile benchmark of A measures no CRUX part.
- The available efficiency data: the parameter counts, the fp16 model size, the peak CUDA memory, and the L4 latency at batch size 1 with and without the cache (Steps 7 and 8). The paper states that no on-device or INT8 measurement exists.

## Step 10: Research Paper and Release

- [ ] Use the 3-seed results from Steps 5 to 8. If you add more seeds, add them to all configurations.
- [ ] Write a LaTeX paper about the methodology, the datasets, the architecture, the experiments, the ablations, the limitations, and the conclusions. Include negative results.
- [ ] Release the code, the configs, the split hashes, the weights, and the data where the licenses permit.
- [ ] Make sure that a clean Colab run of the notebooks gives the reported numbers within the seed variance.

**Carry-over from earlier steps:**
- Give attribution to ProofWriter (CC BY 4.0). Keep the MIT copyright notice of Ettin (Step 0).
- Do not redistribute Super-NaturalInstructions instances. Release the task IDs and hashes only (Step 0).
- Make novelty claims only for the combination of parts (Step 0, `docs/literature_review.md`).
