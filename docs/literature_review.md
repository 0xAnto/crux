# CRUX Literature Review (Step 0)

Scope: this is a short, targeted review. It is not a systematic survey. Each reference below was checked against its arXiv page, the ACL Anthology, or the proceedings page. The "not found" statements mean "not found in this search". They do not prove that no prior work exists.

## 1. Instruction-conditioned hypernetworks

A hypernetwork is a network that generates the weights of another network (Ha et al., 2016). Several NLP papers use text to condition such a network.

- Ye and Ren (2021) generate adapters from a task description. The generated adapters give a model zero-shot transfer to unseen tasks.
- Mahabadi et al. (2021) use a shared hypernetwork to generate adapter weights for many tasks in one model.
- HyperTuning (Phang et al., 2023) uses a hypermodel (HyperT5). The hypermodel reads few-shot examples. It outputs a soft prefix or LoRA weights for a frozen T5 model. It trains in two stages: hyperpretraining, then multi-task fine-tuning.
- HINT (Ivison et al., 2023) encodes the instruction with a pretrained text encoder. It converts the result into parameter-efficient modules. The instruction does not enter the model input again for each example. This saves compute when the same instruction serves many inputs.
- Text-to-LoRA (Charakorn et al., 2025) generates a LoRA adapter from a task description in one forward pass.
- Volk et al. (2023) generate the weights of a classifier from a signature of the input example. The goal is adaptation to unseen domains.

These works use large models (T5 and larger) and task descriptions or few-shot examples. We did not find a work that conditions a low-rank head on formal rule text for a 17M-30M encoder. LoRA (Hu et al., 2021) is the low-rank parametrization that B uses.

## 2. Counterfactual data and consistency training

- Kaushik et al. (2020) ask humans to edit an example so that its label flips. Training on the original and edited examples improves robustness to spurious features.
- Gardner et al. (2020) define contrast sets: small edits that change the label. They use them to measure the local decision boundary of a model.
- Li et al. (2019) and Asai and Hajishirzi (2020) add logic-based consistency losses and logic-guided data augmentation. Elazar et al. (2021) measure and improve paraphrase consistency of pretrained language models.
- Nafar et al. (2024) add logical-constraint losses to transformers on a probabilistic RuleTaker variant.

Counterfactual pairs and consistency losses are established. CRUX differs in one detail: it generates the pairs with a symbolic rule engine, so the label of each pair is verified by program. We did not find a prior work that combines this with a hypernetwork.

## 3. Rule-reasoning datasets and models

- RuleTaker (Clark et al., 2020) shows that transformers can learn to answer questions over facts and natural-language rules.
- ProofWriter (Tafjord et al., 2021) generates implications and proofs over the same data format.
- BoardgameQA (Kazemi et al., 2023) adds contradictory information, rule priorities, and missing information (proved / disproved / unknown).
- DynaGuard and DynaBench (Hoover et al., 2025) cover user-defined policies for guardrail decisions. DynaBench has a 61.5k training set and a 543-example test set. The DynaGuard models have 1.7B, 4B, and 8B parameters and use Qwen3 as the base model. They are much larger than CRUX.
- Super-NaturalInstructions (Wang et al., 2022) gives the held-out instruction-following classification tasks.

## 4. Architecture D: cacheable and permutation-equivariant scoring

Cacheable representations:

- Bi-encoders and cross-encoders trade accuracy for speed. Poly-encoders (Humeau et al., 2020) sit between the two. They precompute the candidate side and add a light attention step.
- ColBERT (Khattab and Zaharia, 2020) encodes query and document separately. It keeps token-level vectors for a late interaction step. The document side is precomputed.
- DeFormer (Cao et al., 2020) removes cross-attention between question and context in the lower layers of a transformer. The system precomputes and caches the context. It loses some accuracy in return. This is the same trade-off that the C+cache ablation measures.

Permutation-equivariant candidate scoring:

- Deep Sets (Zaheer et al., 2017) and Set Transformer (Lee et al., 2019) give the theory and the attention-based modules for set inputs.
- A standard multiple-choice head (for example, SWAG-style, Zellers et al., 2018) already scores each candidate with shared weights. This is permutation-equivariant by design. Zheng et al. (2024) show that decoder LLMs are sensitive to option order. An encoder with a shared per-candidate scorer does not have this problem by construction.

Only an extra cross-candidate step (for example, a Set Transformer block over candidate vectors) adds something beyond the standard head.

## 5. Small encoders

- Ettin (Weller et al., 2026) is an open suite of paired encoders and decoders from 17M to 1B parameters. The models use the same training recipe. Ettin-17M is the CRUX base (A). Ettin-32M is an external baseline.
- ModernBERT (Warner et al., 2025) is a modern encoder with a long context and fast inference. Ettin builds on this kind of recipe.
- DeBERTa-v3 (He et al., 2023) uses ELECTRA-style pretraining with gradient-disentangled embedding sharing. DeBERTa-v3-xsmall is the second external baseline.

Step 1 must check whether the Ettin pretraining data contains RuleTaker or ProofWriter. This review does not check it.

## Novelty of CRUX parts

No table cell below claims a measured result. "Novel" means "no prior work found in this search". It does not mean "better".

| Part | Closest prior work | Novel or not novel | Reason |
|---|---|---|---|
| Rule-conditioned low-rank hypernetwork head (B) | HyperTuning, HINT, Text-to-LoRA, Ye and Ren 2021, Volk et al. 2023 | Not novel (the application is new) | Text-conditioned hypernetworks that output LoRA or adapters exist. We found none for a 17M-30M encoder that makes rule decisions. |
| Counterfactual training + consistency objective (C) | Kaushik et al. 2020, Gardner et al. 2020, Asai and Hajishirzi 2020, Li et al. 2019, Elazar et al. 2021 | Not novel | Counterfactual data and consistency losses are known. Only the engine-verified pairs and the pairing with a hypernetwork are new details. |
| Cacheable state and rule representations (D, part 1) | DeFormer, ColBERT, Poly-encoders, HINT | Not novel alone | Separate encoding with caching is known. HINT already caches instruction-side modules. The C+cache ablation measures the known accuracy cost. |
| Permutation-equivariant candidate scoring (D, part 2) | Deep Sets, Set Transformer, standard multiple-choice heads (SWAG) | Not novel | A shared per-candidate scorer is already equivariant. The extra cross-candidate step is a known set module. |
| The combination at ~17-30M parameters for mobile System-1 decisions | DynaGuard (1.7B-8B), HINT (T5-scale), Ettin | Possibly novel (unproven) | We found no work that joins all parts at this size. The claim holds only if the Step 8 results show reliable gains over A, A-param, and the CF controls. |

## References

- Ha, D., Dai, A., Le, Q. V. (2016). HyperNetworks. arXiv:1609.09106. https://arxiv.org/abs/1609.09106
- Ye, Q., Ren, X. (2021). Learning to Generate Task-Specific Adapters from Task Description. ACL 2021. arXiv:2101.00420. https://arxiv.org/abs/2101.00420
- Karimi Mahabadi, R., Ruder, S., Dehghani, M., Henderson, J. (2021). Parameter-efficient Multi-task Fine-tuning for Transformers via Shared Hypernetworks. ACL 2021. arXiv:2106.04489. https://arxiv.org/abs/2106.04489
- Phang, J., Mao, Y., He, P., Chen, W. (2023). HyperTuning: Toward Adapting Large Language Models without Back-propagation. ICML 2023 (PMLR 202). arXiv:2211.12485. https://arxiv.org/abs/2211.12485
- Ivison, H., Bhagia, A., Wang, Y., Hajishirzi, H., Peters, M. (2023). HINT: Hypernetwork Instruction Tuning for Efficient Zero- and Few-Shot Generalisation. ACL 2023. arXiv:2212.10315. https://arxiv.org/abs/2212.10315
- Charakorn, R., Cetin, E., Tang, Y., Lange, R. T. (2025). Text-to-LoRA: Instant Transformer Adaption. ICML 2025. arXiv:2506.06105. https://arxiv.org/abs/2506.06105
- Volk, T., Ben-David, E., Amosy, O., Chechik, G., Reichart, R. (2023). Example-based Hypernetworks for Multi-source Adaptation to Unseen Domains. Findings of EMNLP 2023. https://doi.org/10.18653/v1/2023.findings-emnlp.610
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685. https://arxiv.org/abs/2106.09685
- Kaushik, D., Hovy, E., Lipton, Z. C. (2020). Learning the Difference that Makes a Difference with Counterfactually-Augmented Data. ICLR 2020. arXiv:1909.12434. https://arxiv.org/abs/1909.12434
- Gardner, M., Artzi, Y., Basmova, V., Berant, J., Bogin, B., et al. (2020). Evaluating Models' Local Decision Boundaries via Contrast Sets. Findings of EMNLP 2020. arXiv:2004.02709. https://arxiv.org/abs/2004.02709 . DOI: https://doi.org/10.18653/v1/2020.findings-emnlp.117
- Li, T., Gupta, V., Mehta, M., Srikumar, V. (2019). A Logic-Driven Framework for Consistency of Neural Models. EMNLP 2019. arXiv:1909.00126. https://arxiv.org/abs/1909.00126
- Asai, A., Hajishirzi, H. (2020). Logic-Guided Data Augmentation and Regularization for Consistent Question Answering. ACL 2020. arXiv:2004.10157. https://arxiv.org/abs/2004.10157
- Elazar, Y., Kassner, N., Ravfogel, S., Ravichander, A., Hovy, E., Schutze, H., Goldberg, Y. (2021). Measuring and Improving Consistency in Pretrained Language Models. TACL. arXiv:2102.01017. https://arxiv.org/abs/2102.01017
- Nafar, A., Venable, K. B., Kordjamshidi, P. (2024). Teaching Probabilistic Logical Reasoning to Transformers. Findings of EACL 2024. arXiv:2305.13179. https://arxiv.org/abs/2305.13179
- Clark, P., Tafjord, O., Richardson, K. (2020). Transformers as Soft Reasoners over Language. IJCAI 2020. arXiv:2002.05867. https://arxiv.org/abs/2002.05867
- Tafjord, O., Dalvi Mishra, B., Clark, P. (2021). ProofWriter: Generating Implications, Proofs, and Abductive Statements over Natural Language. Findings of ACL 2021. arXiv:2012.13048. https://arxiv.org/abs/2012.13048
- Kazemi, M., Yuan, Q., Bhatia, D., Kim, N., Xu, X., Imbrasaite, V., Ramachandran, D. (2023). BoardgameQA: A Dataset for Natural Language Reasoning with Contradictory Information. NeurIPS 2023 (Datasets and Benchmarks). arXiv:2306.07934. https://arxiv.org/abs/2306.07934
- Hoover, M., Baherwani, V., Jain, N., Saifullah, K., Vincent, J., Jain, C., Kazemi Rad, M., Bruss, C. B., Panda, A., Goldstein, T. (2025). DynaGuard: A Dynamic Guardian Model With User-Defined Policies. arXiv preprint, arXiv:2509.02563. https://arxiv.org/abs/2509.02563
- Wang, Y., Mishra, S., Alipoormolabashi, P., et al. (2022). Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks. EMNLP 2022. arXiv:2204.07705. https://arxiv.org/abs/2204.07705
- Humeau, S., Shuster, K., Lachaux, M.-A., Weston, J. (2020). Poly-encoders: Transformer Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scoring. ICLR 2020. arXiv:1905.01969. https://arxiv.org/abs/1905.01969
- Khattab, O., Zaharia, M. (2020). ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. SIGIR 2020. arXiv:2004.12832. https://arxiv.org/abs/2004.12832
- Cao, Q., Trivedi, H., Balasubramanian, A., Balasubramanian, N. (2020). DeFormer: Decomposing Pre-trained Transformers for Faster Question Answering. ACL 2020. arXiv:2005.00697. https://arxiv.org/abs/2005.00697
- Zaheer, M., Kottur, S., Ravanbakhsh, S., Poczos, B., Salakhutdinov, R., Smola, A. (2017). Deep Sets. NeurIPS 2017. arXiv:1703.06114. https://arxiv.org/abs/1703.06114
- Lee, J., Lee, Y., Kim, J., Kosiorek, A. R., Choi, S., Teh, Y. W. (2019). Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks. ICML 2019. arXiv:1810.00825. https://arxiv.org/abs/1810.00825
- Zellers, R., Bisk, Y., Schwartz, R., Choi, Y. (2018). SWAG: A Large-Scale Adversarial Dataset for Grounded Commonsense Inference. EMNLP 2018. arXiv:1808.05326. https://arxiv.org/abs/1808.05326
- Zheng, C., Zhou, H., Meng, F., Zhou, J., Huang, M. (2024). Large Language Models Are Not Robust Multiple Choice Selectors. ICLR 2024. arXiv:2309.03882. https://arxiv.org/abs/2309.03882
- Weller, O., Ricci, K., Marone, M., Chaffin, A., Lawrie, D., Van Durme, B. (2026). Seq vs Seq: An Open Suite of Paired Encoders and Decoders (Ettin). ICLR 2026. arXiv:2507.11412. https://arxiv.org/abs/2507.11412
- Warner, B., Chaffin, A., Clavie, B., Weller, O., et al. (2025). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference (ModernBERT). ACL 2025. arXiv:2412.13663. https://arxiv.org/abs/2412.13663 . DOI: https://doi.org/10.18653/v1/2025.acl-long.127
- He, P., Gao, J., Chen, W. (2023). DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing. ICLR 2023. arXiv:2111.09543. https://arxiv.org/abs/2111.09543
