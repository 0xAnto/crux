# Ettin Contamination Check (Step 1)

Date of check: 2026-10-14.

Scope: this is a documentation search. It is not a scan of the 2T-token corpus. "Not found in the documentation" means that the keyword is absent from the pages listed below. It does not prove that the data is absent from the corpus.

Model scope: Ettin-17M and Ettin-32M use the same recipe and the same data (Weller et al., arXiv 2507.11412). We read the 17M card, the data cards, the paper, and the repo.

## Data mixture (from the Ettin paper, Table 2, and the data cards)

| Phase | Tokens | Instruction source |
|---|---|---|
| Pre-training | 1,704.7B | Tulu Flan, 16.6B (1.0%) |
| Mid-training / context extension | 249.3B | Tulu Flan, 2.4B (1.0%) |
| Decay | 76.3B in the paper, 81.6B on the card (the text says 50B) | Tulu Flan, 4.1B (5.0%) |

Other sources: DCLM, CC Head, CC News, StarCoder, Reddit, PeS2o, Arxiv, StackExchange, Open-Web-Math, Algebraic StackExchange, Wikipedia, Dolma Books, textbooks, code repos, and the Dolmino filtered sets (DCLM, math, StackExchange). The paper says the instruction data appears in all phases ("trained with instructions during pre-training", Appendix F).

The `tulu_flan` folder comes from Dolma v1.7. The repo script `pretraining/data_processing/bin/dolma_mds_all.sh` converts the Dolma tag `tulu_flan`. The Dolma v1.7 URL list contains `olmo-data.org/dolma-v1_7/tulu_flan/`. The Dolma card describes this source as the "Flan Collection", reproduced with the original FLAN v2 code (Dettmers et al., 2023): 52.1M documents, 16.5B tokens, cutoff Feb 2023. The token count matches the Ettin count (16.6B). The Dolma card does not use the name "tulu_flan". We infer the link from the URL list and the token counts.

## Findings

| # | Question | Verdict | Source |
|---|---|---|---|
| 1 | RuleTaker or ProofWriter in the Ettin paper, model card, GitHub repo, or the three HF data cards | not found in the documentation | https://arxiv.org/abs/2507.11412 , https://huggingface.co/jhu-clsp/ettin-encoder-17m , https://github.com/JHU-CLSP/ettin-encoder-vs-decoder , https://huggingface.co/datasets/jhu-clsp/ettin-pretraining-data (and `-extension-data`, `-decay-data`) |
| 2 | Decontamination against evaluation sets in the Ettin paper | not found in the documentation (the word does not appear) | https://arxiv.org/html/2507.11412v2 |
| 3 | File or folder names with "ruletaker", "proofwriter", "allenai", "natural-instructions" in the HF data repos | not found in the documentation. The top-level folders are source names only. The `tulu_flan/` folder holds MDS shards. We did not list every shard. | https://huggingface.co/datasets/jhu-clsp/ettin-pretraining-data/tree/main |
| 4 | RuleTaker in the upstream Flan Collection | found. The NIv2 part lists `task966_ruletaker_fact_checking_based_on_given_context` and `task967_ruletaker_incorrect_fact_generation_based_on_given_paragraph`. Both tasks are in the Super-NaturalInstructions default train split. | https://github.com/google-research/FLAN/blob/main/flan/v2/flan_collection_info.csv , https://github.com/allenai/natural-instructions/tree/master/tasks , https://github.com/allenai/natural-instructions/blob/master/splits/default/train_tasks.txt |
| 5 | RuleTaker in the Ettin `tulu_flan` data | unclear. The Dolma card does not list tasks and says the data is "sampled to balance different Flan subsets". We do not know if tasks 966 and 967 remain. | https://huggingface.co/datasets/allenai/dolma |
| 6 | ProofWriter in the Flan Collection (Flan 2021, T0, NIv2, CoT, dialog) or in the Super-NaturalInstructions task files | not found in the documentation | same as row 4 |
| 7 | ProofWriter in the Ettin data | not found in the documentation | same as row 1 |
| 8 | Super-NaturalInstructions in the Flan Collection | found. The Flan catalog lists 115 of the 119 task IDs of the SNI default held-out test split. It also lists 681 of 756 train task IDs. We did not check which of these tasks survive in the Ettin `tulu_flan` data. | https://github.com/google-research/FLAN/blob/main/flan/v2/flan_collection_info.csv , https://github.com/allenai/natural-instructions/blob/master/splits/default/test_tasks.txt |
| 9 | Super-NaturalInstructions in the Ettin data | unclear (see rows 5 and 8) | same as row 5 |
| 10 | BoardgameQA or DynaBench | not found in the documentation. The Flan catalog and the SNI task files do not name them. The Flan data in Dolma has a cutoff of Feb 2023, which is before both datasets. We did not check web-crawl sources. | https://huggingface.co/datasets/allenai/dolma |

Notes:
- OLMo 2 "Dolmino" also has a Flan source (17.0B tokens, 57.3M documents). Ettin names its later-phase instruction folder `tulu_flan`, the same as in phase 1. The cards do not say which Flan source feeds phases 2 and 3. This is part of row 5. Source: https://huggingface.co/datasets/allenai/dolmino-mix-1124 .
- The web sources (DCLM, CC Head, Reddit, StarCoder, Arxiv) can contain RuleTaker or ProofWriter text. The RuleTaker and ProofWriter data is public (GitHub and Hugging Face). The documentation does not exclude this. We did not check the DCLM or Dolma decontamination lists.

## Consequence for CRUX

- Step 1: a RuleTaker accuracy above the published value can come from the pretraining. Do not read it as a pipeline result alone.
- Step 4: add a measured overlap check. Run the n-gram overlap test of the frozen test split (ProofWriter depth 5, Birds-Electricity, ParaRules) against the public Dolma v1.7 `tulu_flan` shards (about 16.5B tokens). This scan is feasible. Also test the RuleTaker examples inside the SNI tasks 966 and 967. Record the result and the hash of the test split in the same file.
- Step 4: if the overlap check finds a test item, remove the item before the freeze. Report the number of removed items.
- Step 8: all of A, B, C, and D start from the same Ettin weights. A possible contamination affects them in the same way. Paired differences between them are less at risk. Absolute accuracy, and the comparison with DeBERTa-v3-xsmall, are at risk. State this in the limitations.
- SNI held-out tasks (evaluation only): the Flan catalog lists most of them. Check the overlap before you use them. Report them as "possibly seen in pretraining" if the check is not done.
- BoardgameQA and DynaBench: no action is needed from this check. Keep the web-crawl caveat in the limitations.
