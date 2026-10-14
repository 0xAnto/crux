# Data and model licenses

Date of all checks: 2026-10-14.
This file records the license of each data set and each model weight that CRUX uses.
Each license comes from an official source. A mirror label (for example a third-party Hugging Face card) is not a source.
This file is not legal advice.

## Datasets

| Name | License | Official source | Checked | Research use | Redistribution of data or derived data | Notes |
|---|---|---|---|---|---|---|
| RuleTaker | Apache-2.0 | Ai2 data page (archived snapshot 2023-02-08, "License: Apache 2.0"): https://allenai.org/data/ruletaker. Code repo: https://github.com/allenai/ruletaker (Apache-2.0). | 2026-10-14 | Yes | Yes. Keep the license notice. | The live Ai2 page now redirects to the GitHub repo. The repo README states the Apache-2.0 license for the code only. The data zip has no license file. The data license comes from the archived Ai2 page. |
| ProofWriter | CC BY (the Ai2 page links CC BY 4.0) | Ai2 data page (archived snapshot 2023-02-08, "License: CC BY", link to creativecommons.org/licenses/by/4.0): https://allenai.org/data/proofwriter. Data: https://aristo-data-public.s3.amazonaws.com/proofwriter/proofwriter-dataset-V2020.12.3.zip | 2026-10-14 | Yes | Yes, with attribution. | The live Ai2 page now redirects to the data zip. The zip README has no license text. The data license comes from the archived Ai2 page. |
| BoardgameQA | CC BY (the paper does not state a version) | Paper data statement: "Available at https://storage.googleapis.com/gresearch/BoardgameQA/BoardgameQA.zip. License: CC BY." https://arxiv.org/abs/2306.07934 (NeurIPS 2023). The zip URL returns HTTP 200. | 2026-10-14 | Yes | Yes, with attribution. | I found no official GitHub repo with a LICENSE file. The `tasksource/Boardgame-QA` card on Hugging Face is a third-party mirror. Do not cite it. |
| DynaBench (DynaGuard paper, 2025) | MIT | Dataset card `montehoover/DynaBench`, license field `mit`. The official DynaGuard collection links to it: https://huggingface.co/collections/tomg-group-umd/dynaguard-68af4d916ae81d06ef774523. Paper: https://arxiv.org/abs/2509.02563. Code: https://github.com/montehoover/DynaGuard (MIT). | 2026-10-14 | Yes | Yes, for the DynaBench test set (543 handcrafted examples). | This is not the Facebook "Dynabench" platform. The card is in the account of the first author. The paper has no license statement for the data. The `DynaBenchSafetyMix` subset mixes WildGuard, BeaverTails, ToxicChat, and Aegis 2.0. I did not check the licenses of these upstream sets. CRUX does not use that subset. Use only the `DynaBench` subset, `test` split. |
| Super-NaturalInstructions (natural-instructions) | Apache-2.0 for the repo data except task instances. Each task instance set has its own license in the task field `Instance License`. | https://github.com/allenai/natural-instructions (README: "All the data here (except the instances of each task) are released under Apache-2.0 license." and "The instances of each tasks are subject to the license under which the original dataset was released.") | 2026-10-14 | Yes for evaluation. See the per-task notes. | Task definitions: yes (Apache-2.0). Task instances: only as the per-task license allows. Do not redistribute the instances. Publish task IDs and hashes instead. | See the next section. |
| Super-NaturalInstructions official held-out split | Apache-2.0 (split files are part of the repo data) | https://github.com/allenai/natural-instructions/tree/master/splits/default (`test_tasks.txt`: 119 tasks, English track) | 2026-10-14 | Yes | Yes for the task list. | Use `splits/default/test_tasks.txt`. The split has no license of its own other than the repo license. |
| ShARC | Not researched | Not applicable | Not applicable | Excluded | Excluded | The project excludes ShARC (steps.md, Step 0): it has no known license, and published work reports spurious patterns in its labels. |

## Super-NaturalInstructions: per-task licenses of the 119 held-out tasks

I read the `Instance License` field of each of the 119 tasks in `splits/default/test_tasks.txt`.

| License value | Number of tasks |
|---|---|
| Apache 2.0 | 22 |
| MIT | 19 |
| Unknown | 16 |
| CC BY 4.0 | 13 |
| CC BY-SA 4.0 | 12 |
| CC BY-NC 4.0 | 9 |
| CC BY-NC-SA 4.0 | 5 |
| OANC | 4 |
| BSD 2-Clause | 3 |
| CC BY-NC-SA 3.0 | 2 |
| Microsoft Research Data License Agreement | 2 |
| Other single-task licenses (BSD, CC BY-SA, CC BY-SA 3.0, CC BY-SA with GFDL, CC BY-NC 2.0, CC0 1.0, CCG, Amazon, Quora ToS, RecipeNLG terms, University of Washington Academic License, EU copyright notice) | 13 |

Notes:

- 17 tasks have a non-commercial license (CC BY-NC or CC BY-NC-SA). Non-commercial terms allow research evaluation. They do not allow commercial use.
- 16 tasks have the license value "Unknown". I cannot confirm research use for these tasks. See "Decisions".
- CRUX uses only the classification tasks. The classification subset is not yet defined. Compute the license list again for that subset when you define it.
- The task files carry the license field. Copy it into the CRUX evaluation manifest for every task.

## Model weights

| Name | License | Official source | Checked | Research use | Redistribution | Notes |
|---|---|---|---|---|---|---|
| Ettin-17M encoder (`jhu-clsp/ettin-encoder-17m`) | MIT | https://huggingface.co/jhu-clsp/ettin-encoder-17m (model card, JHU CLSP organization). Repo: https://github.com/JHU-CLSP/ettin-encoder-vs-decoder (MIT). | 2026-10-14 | Yes | Yes. Keep the copyright notice. | The license covers the weights as the model card states. The repo page gives no license for the pretraining data. |
| Ettin-32M encoder (`jhu-clsp/ettin-encoder-32m`) | MIT | https://huggingface.co/jhu-clsp/ettin-encoder-32m (model card, JHU CLSP organization). | 2026-10-14 | Yes | Yes. Keep the copyright notice. | Same notes as Ettin-17M. |
| DeBERTa-v3-xsmall (`microsoft/deberta-v3-xsmall`) | MIT | https://huggingface.co/microsoft/deberta-v3-xsmall (model card, Microsoft organization). Repo: https://github.com/microsoft/DeBERTa (MIT). | 2026-10-14 | Yes | Yes. Keep the copyright notice. | None. |

## Decisions

- REMOVE: none. No benchmark has a license that forbids research use.
- RuleTaker and ProofWriter: CRUX accepts the archived Ai2 page (2023-02-08) as the official source. The live Ai2 pages no longer show the license.
- Super-NaturalInstructions: CRUX excludes the 16 held-out tasks with license "Unknown" from the evaluation. Recompute the count when Step 4 defines the classification subset.
- Super-NaturalInstructions: the CRUX release does not include task instances. It includes the task IDs and hashes only.

## Project license

- Code, trained weights, and the synthetic counterfactual dataset (Step 3): Apache-2.0. See `LICENSE`.
- Paper: CC BY 4.0, only if the venue requires it.
- Compatibility: the training inputs permit Apache-2.0 for the outputs. RuleTaker is Apache-2.0. Ettin is MIT. ProofWriter is CC BY 4.0. The release must include attribution for ProofWriter and the MIT copyright notice for Ettin (in `NOTICE` or in the model card).
- The evaluation-only benchmarks do not change the license of the weights. CRUX does not train on them.
