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

- [ ] Make train, validation, and test splits. The test split contains rule templates, compositions, vocabulary, and paraphrases that do not occur in the train split.
- [ ] Hold out by reasoning depth and by domain: use ProofWriter depth 5, Birds-Electricity, and ParaRules only for tests.
- [ ] Check for leakage automatically: no shared template IDs, no near-duplicate texts (n-gram overlap) between train and test.
- [ ] Record a hash for each split file, then freeze the splits.
- [ ] Implement these metrics: accuracy, counterfactual paired accuracy, paraphrase consistency, calibration (ECE), option-order invariance, and unseen-rule accuracy.

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

- [ ] Train the official A on the frozen train split (3 seeds).
- [ ] Extend Ettin-17M with an instruction-conditioned rank-8 hypernetwork. The hypernetwork generates low-rank parameters for the decision head.
- [ ] Ablation: the hypernetwork also generates low-rank adapters for the top K encoder layers.
- [ ] Control **A-param**: give A an unconditioned LoRA with the same number of trainable parameters as B. This control shows if the conditioning helps, or only the extra parameters help.
- [ ] Train with the same data and compute budget as A. Compare the accuracy, the generalization, the latency, and the parameter count on the validation split.

**Carry-over from earlier steps:**
- Set `budget.training_steps` in `configs/protocol.yaml` before you train A. Use the same value for all configurations (Step 0).
- Do not use lr 1e-4 for Ettin-17M. The dev loss is lowest near epoch 6 and then rises. Include lr values near 3e-5 in the 6-trial search (Step 1).
- Report the majority-class baseline with every accuracy. In ProofWriter OWA depth 3, "Unknown" is 44.5% of the train records (Step 2).
- Training uses 32 records per batch in all modes (`FixedBatchSampler`). Equal steps then give equal data (Step 2).
- Token limits (`configs/step2_data.yaml`): `max_length` 640, `max_instruction_length` 448, `max_state_length` 256. No synthetic record is truncated (Step 3 report: a sample of 2,000 records).

## Step 6: Architecture C: Counterfactual Learning

- [ ] Extend B with paired counterfactual training, the supervised decision loss, and a consistency loss for logically equivalent instructions.
- [ ] Control **A+CF**: train A on the same counterfactual data. This control separates the effect of the data from the effect of the architecture.
- [ ] Ablation **B+CF**: train B on the counterfactual data without the consistency loss. This ablation separates the effect of the data from the effect of the loss.
- [ ] Compare C, B, A+CF, and B+CF on the validation split.

## Step 7: Architecture D: CRUX

- [ ] Extend C with independently encoded, cacheable `state` and `instruction` representations, and a permutation-equivariant candidate-scoring module.
- [ ] Implement shared computation, so that one cached `state` serves many questions.
- [ ] Ablation **C+cache**: caching only. Independent encoding can decrease accuracy compared with joint encoding. Measure this trade-off.
- [ ] Ablation **C+equivariant**: candidate scoring only.
- [ ] Evaluate unseen-rule generalization, option-order invariance, memory consumption, and inference latency (with and without a cache hit).

**Carry-over from earlier steps:**
- Binary options use one canonical order in all sources (positive label first, then negative, then "Unknown"). Shuffle the option order when you measure order invariance (Step 2).
- Multi-choice pairs with a rule edit occur only in compositions with `pri`. Validation has only c23 (31 units), so report the validation results for these pairs as one-composition results. Count these pairs per block in the report (Step 3).

## Step 8: Benchmark All Models

- [ ] Evaluate A, A-param, A+CF, B, B+CF, C, D, the two ablations of D, and the external baselines on the frozen test split. Evaluate also on RuleTaker, ProofWriter, BoardgameQA, DynaBench, and the Super-NaturalInstructions held-out classification tasks (only the benchmarks that pass the Step 0 license check).
- [ ] Use one evaluation script and one protocol for all models.
- [ ] Generate comparison tables and plots for accuracy, paired counterfactual accuracy, calibration, generalization, and latency. Write the better direction in each table title, for example "(higher is better)".
- [ ] Add 95% CIs and the paired bootstrap test for each comparison with A.
- [ ] Apply the criteria from Step 0. Report all results, including the comparisons where the CI includes 0.

**Carry-over from earlier steps:**
- Limitations: A, B, C, and D share the Ettin initialization, so a contamination affects them equally. DeBERTa-v3-xsmall has different pretraining data (Step 1).
- DynaBench has only 543 test examples, so its CIs are wide (Step 0).
- Long inputs: DynaBench has up to 5,486 joint tokens. With the Step 2 limits (`max_length` 640, `max_instruction_length` 448, `max_state_length` 256), joint mode truncates 350 of 543 DynaBench records (rules 314, state 136), and separate mode truncates 397. BoardgameQA: joint mode with options truncates 772 of 37,500 records, and separate mode truncates 1,832 (rules 1,807, state 45). The question stays whole. Set larger evaluation limits (Ettin supports about 8K tokens). Use the same limits for all models (Step 2).
- Shuffle the option order when you measure order invariance (Step 2).
- Report each edit type split by the orig label. The mix of edit types depends on the answer: for binary units, `condition_negate` is 0.65 of the units that start at Unknown and 0.25 of those that start at True or False, and `rule_remove` is 0.001 and 0.14. `fact_remove` from Unknown has the flip share 0.29 (a depth-0 Unknown is always a stay) (Step 3).
- Do not report per-type results of `exception_remove` and `priority_swap` for val (22 and 4 units) (Step 3).

## Step 9: Optimize for Mobile

- [ ] Select the architecture with the best validation result, not the best test result.
- [ ] Export it to ONNX. If the hypernetwork generates weights at runtime, check that the export keeps this computation, or pre-compile the parameters for each fixed instruction.
- [ ] Apply INT8 quantization. Compare the INT8 and FP32 versions for accuracy loss.
- [ ] Measure model size, peak RAM, and latency on Colab.
- [ ] Build an Android benchmark with ONNX Runtime Mobile. Measure on the device named in Step 0, and record the device, the OS version, and the thread count.

**Carry-over from earlier steps:**
- Measure on the Google Pixel 8 only. Keep the same Android OS version for all measurements. Record the OS version and the thread count (Step 0).

## Step 10: Research Paper and Release

- [ ] Use the 3-seed results from Steps 5 to 8. If you add more seeds, add them to all configurations.
- [ ] Write a LaTeX paper about the methodology, the datasets, the architecture, the experiments, the ablations, the limitations, and the conclusions. Include negative results.
- [ ] Release the code, the configs, the split hashes, the weights, and the data where the licenses permit.
- [ ] Make sure that a clean Colab run of the notebooks gives the reported numbers within the seed variance.

**Carry-over from earlier steps:**
- Give attribution to ProofWriter (CC BY 4.0). Keep the MIT copyright notice of Ettin (Step 0).
- Do not redistribute Super-NaturalInstructions instances. Release the task IDs and hashes only (Step 0).
- Make novelty claims only for the combination of parts (Step 0, `docs/literature_review.md`).
