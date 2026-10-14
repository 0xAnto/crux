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

- [ ] Train Ettin-17M (`jhu-clsp/ettin-encoder-17m`) on RuleTaker with fp16 mixed precision.
- [ ] Use a head that matches the labels: 2 classes for closed-world data, 3 classes (True/False/Unknown) for open-world data.
- [ ] Compare the accuracy with the published RuleTaker results. This step only checks the training pipeline. The official A results come from Step 5, on the frozen splits.
- [ ] Save the checkpoints, the configurations, and the training metrics.
- [x] Check for contamination: search the Ettin pretraining data documentation for RuleTaker and ProofWriter. Record the result.

**Done so far:**
- Contamination check: `docs/contamination.md`. RuleTaker and ProofWriter are not in the Ettin documentation. The Flan data in Ettin can contain the RuleTaker tasks task966 and task967 and 115 of the 119 Super-NaturalInstructions held-out tasks. This is not confirmed.
- Step 1 trains and evaluates on RuleTaker dev only. It does not read RuleTaker test.
- Pipeline: `crux/train_step1.py`, `crux/data/ruletaker.py`, `configs/step1_ruletaker.yaml`, `notebooks/step1_smoke.ipynb`. Data: RuleTaker `original/depth-5` (closed world, 2 classes) from the official Ai2 zip. The config pins the SHA-256 of the zip. The loader refuses the test split.
- The CPU smoke run (5 steps) passes end to end. Its output is deleted. It is not a result.
- Open: the Colab T4 run. Record the dev accuracy in `docs/step1_report.md`. Reference: 99.2% test accuracy (Clark et al. 2020, RoBERTa-large). The lr 1e-4, batch 32, and 3 epochs are not tested on a GPU.
- Carry-over to Step 4: the 3-class open-world data comes from the ProofWriter release. It has a different file format, so the loader needs an adaptation.

## Step 2: Unified Decision Schema

- [ ] Build a preprocessing pipeline that converts each dataset into `{state, instruction, options, target}` for the binary, multi-choice, and scoring task types.
- [ ] Keep `state` (facts and evidence) and `instruction` (rules and question) as separate fields. Architecture D encodes and caches these fields independently.
- [ ] Support a variable number of options per example.
- [ ] Implement tokenization, dynamic batching, dataset validation, and reusable PyTorch DataLoaders.

## Step 3: Counterfactual Dataset Generator

- [ ] Build a deterministic synthetic generator that supports logical rules, negations, thresholds, exceptions, priorities, and multi-step dependencies.
- [ ] Give each example a template ID, a composition ID, and a vocabulary ID. Step 4 uses these IDs to make the splits.
- [ ] Generate 100K labeled examples with matched counterfactual pairs. A pair changes one fact or one rule, and the label changes or stays the same as the rule engine computes.
- [ ] Include multi-choice examples with 2 to 8 options. Without these, Step 7 cannot measure the effect of candidate scoring.
- [ ] Generate paraphrases of the instructions that are logically equivalent.
- [ ] Validate every label with an executable symbolic rule engine. Reject an example if the engine and the generator disagree.

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

## Step 5: Architecture A (Official) and Architecture B: Rule Compiler

- [ ] Train the official A on the frozen train split (3 seeds).
- [ ] Extend Ettin-17M with an instruction-conditioned rank-8 hypernetwork. The hypernetwork generates low-rank parameters for the decision head.
- [ ] Ablation: the hypernetwork also generates low-rank adapters for the top K encoder layers.
- [ ] Control **A-param**: give A an unconditioned LoRA with the same number of trainable parameters as B. This control shows if the conditioning helps, or only the extra parameters help.
- [ ] Train with the same data and compute budget as A. Compare the accuracy, the generalization, the latency, and the parameter count on the validation split.

**Carry-over from earlier steps:**
- Set `budget.training_steps` in `configs/protocol.yaml` before you train A. Use the same value for all configurations (Step 0).

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

## Step 8: Benchmark All Models

- [ ] Evaluate A, A-param, A+CF, B, B+CF, C, D, the two ablations of D, and the external baselines on the frozen test split. Evaluate also on RuleTaker, ProofWriter, BoardgameQA, DynaBench, and the Super-NaturalInstructions held-out classification tasks (only the benchmarks that pass the Step 0 license check).
- [ ] Use one evaluation script and one protocol for all models.
- [ ] Generate comparison tables and plots for accuracy, paired counterfactual accuracy, calibration, generalization, and latency. Write the better direction in each table title, for example "(higher is better)".
- [ ] Add 95% CIs and the paired bootstrap test for each comparison with A.
- [ ] Apply the criteria from Step 0. Report all results, including the comparisons where the CI includes 0.

**Carry-over from earlier steps:**
- Limitations: A, B, C, and D share the Ettin initialization, so a contamination affects them equally. DeBERTa-v3-xsmall has different pretraining data (Step 1).
- DynaBench has only 543 test examples, so its CIs are wide (Step 0).

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
