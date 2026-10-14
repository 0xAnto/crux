# CRUX: Compact System One Decision Model

**Project Goal:** Build a novel, open-source, mobile-efficient System 1 decision model inspired by TypeSafe AI's Jev, and to test whether it improves decision accuracy, unseen-rule generalization, and inference efficiency at approximately 17M to 30M parameters.

**Research Hypothesis:** Explicit rule-conditioned computation combined with counterfactual learning can enable small neural networks to generalize to unfamiliar decision tasks more effectively than conventional transformer-based classifiers.

**Four Architectures to Investigate:**

* **A: Baseline:** Ettin-17M encoder with a standard decision head.
* **B: Rule Compiler:** Baseline with an instruction-conditioned low-rank hypernetwork.
* **C: Counterfactual Learning:** Architecture B with counterfactual training and consistency objectives.
* **D: CRUX:** Architecture C with independently cacheable state/rule representations and permutation-equivariant candidate scoring.

**Controls:** A-param (A with the same trainable parameters as B, without conditioning), A+CF (A with the counterfactual data of C), B+CF (B with counterfactual data, without the consistency loss), and separate ablations of the two parts of D. See `steps.md`.

**Research Objectives:**

1. Compare all four architectures under fair, reproducible conditions.
2. Improve generalization to unseen rules, instructions, and decision scenarios.
3. Achieve low-latency, offline mobile inference with minimal memory consumption.
4. Evaluate against open-source baselines (fine-tuned Ettin-32M and DeBERTa-v3-xsmall). Jev has no public weights and only self-reported results, so compare against Jev only through API access on public tasks, and label these results as not reproducible.
5. Produce a reproducible research paper, open-source implementation, and trained model weights.

**Technology:** Python, PyTorch, Hugging Face, Google Colab (T4 GPU), ONNX, and Android-compatible inference runtimes.

**Data:** RuleTaker, ProofWriter, BoardgameQA, DynaBench, Super-NaturalInstructions held-out classification tasks (evaluation only), and programmatically verified synthetic counterfactual examples. Confirm each license from the official source and record it in `DATA_LICENSES.md` before use.

**Execution Instructions:** Implement the project incrementally following Steps 0 through 10 in `steps.md`. Maintain a modular codebase, reproducible experiments, fixed evaluation splits, training logs, checkpoints, and automated tests. Ensure experiments are runnable within Google Colab constraints. Avoid data leakage, document dataset licenses, and use identical evaluation protocols for all models. Never fabricate results or assume superiority without empirical evidence.

**Final Deliverables:** Four trained experimental architectures, evaluation and ablation reports, the best optimized mobile model, reproducible code and datasets where redistribution is permitted, and a publication-ready LaTeX research paper.

**Success Criterion:** Show statistically reliable improvements in held-out rule generalization and counterfactual decision accuracy over A, with practical mobile inference performance. `configs/protocol.yaml` (Step 0) defines the seeds (3 per configuration) and the statistical test before training starts. An improvement is reliable only if its 95% paired bootstrap CI excludes 0. There are no numeric targets before the first measurements.

