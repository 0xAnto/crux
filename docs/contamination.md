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

## Step 4 measured overlap

Date of scan: 2026-10-16. Code: `crux/contamination.py`. Command:
`python -m crux.contamination scan --name step4_test --test data/splits/test.jsonl --sni task966 task967 --source dolma`.

- Corpus: Dolma v1.7 `tulu_flan`, the 66 files `tulu_flan-0000.json.gz` to `tulu_flan-0065.json.gz` of the URL list
  `https://huggingface.co/datasets/allenai/dolma/resolve/main/urls/v1_7.txt` (29,144,203,991 bytes, 11,018,642,377 word tokens).
  All 66 files are scanned. The scan streams the files and does not keep them.
- Settings: word 8-grams of lowercase `[a-z0-9]+` tokens of `state + " " + instruction`. A record is flagged if more than 0.10 of its distinct
  8-grams occur in the corpus. These are `leakage.ngram_n` and `leakage.max_overlap` of `configs/protocol.yaml`.
- Scanned test split: the first freeze, 43,394 records, SHA-256 `baa1699b1e41ecfef6b71a167c5e6267f187961fdf3ab80461b736cf79a0ac93`.

Overlap with `tulu_flan` per subset (share of the 8-grams of a record found in the corpus; lower is better).

| Subset | Records | Records with a shared 8-gram | Flagged (> 0.10) | Max share | p99 share |
|---|---|---|---|---|---|
| synthetic test block | 10,086 | 0 | 0 | 0.0000 | 0.0000 |
| ProofWriter depth 5 | 20,030 | 17,290 | 1,173 | 0.2013 | 0.1497 |
| Birds-Electricity | 5,270 | 0 | 0 | 0.0000 | 0.0000 |
| ParaRules | 8,008 | 332 | 0 | 0.0106 | 0.0085 |
| SNI task966 (RuleTaker, not in the test split) | 300 | 300 | 300 | 1.0000 | 1.0000 |
| SNI task967 (RuleTaker, not in the test split) | 296 | 295 | 291 | 1.0000 | 1.0000 |

Findings:

- Row 5 of the findings table is now resolved: the RuleTaker examples of the SNI tasks 966 and 967 occur verbatim in `tulu_flan`. Thus Ettin saw RuleTaker text in pretraining.
- The flagged depth-5 records are not copies (the maximum share is 0.20). The shared 8-grams are RuleTaker sentence frames with the RuleTaker names,
  for example "the bald eagle then the bald eagle is". ProofWriter uses the same generator and the same names as RuleTaker.
  The 1,173 flagged records come from 67 theories.
- The synthetic test block and Birds-Electricity share no 8-gram with the corpus. ParaRules shares a few (max share 0.0106).

Action (the rule of the section above): the split build removes the 1,173 flagged records from the test split before the freeze. The ids are in
`configs/step4_contamination_removed.txt`. The frozen test split has 42,221 records (`depth5`: 18,857), SHA-256
`fbe0cfcae2b99f98e7a7478e0e3945a29e72ac250feceef7b8ab452b7b4aff58`. The removed records are a subset of the scanned records, so the scan is
valid for the rest: 16,117 of the 18,857 remaining `depth5` records share at least one 8-gram, with a maximum share of 0.10.
Step 8 reports `depth5` with this caveat: the frames and names of its text occur in the pretraining data of all Ettin-based models.

Not scanned: the Super-NaturalInstructions held-out benchmark (`data/splits/benchmarks/supernat.jsonl`). The Flan catalog lists 115 of its 119 tasks.
Step 8 reports it as "possibly seen in pretraining", or a later scan with the same command (`--test data/splits/benchmarks/supernat.jsonl`) measures it.
