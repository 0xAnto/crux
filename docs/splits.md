# Evaluation Dataset and Frozen Splits (Step 4)

Code: `crux/splits.py`. Settings: `configs/step4_splits.yaml`. Frozen facts: `configs/splits_manifest.json`.
The split files are in `data/splits/` (git-ignored). The manifest is in git. It holds the SHA-256 and the counts of each
file, the leakage report, and the baselines of this page.

Build: `python -m crux.splits build`. Check the files against the manifest: `python -m crux.splits verify`.
Count the Unknown-license tasks of Super-NaturalInstructions again: `python -m crux.splits sni-recount`.
Load a split in code: `crux.splits.load_split(name)`. The loader checks the SHA-256 against the manifest
and, for `train`, `validation`, and `test`, against the hash that `protocol.yaml` pre-registers.
It refuses `test` unless the call sets `allow_test=True`. Only Step 8 sets it.

## Files

Record counts and SHA-256 of the frozen files. All files use the schema of `docs/schema.md`, one record per line, sorted by `id`.

| File | Records | SHA-256 |
|---|---|---|
| `train.jsonl` | 75,646 | `bd8cc1141ba3054c94c6e2554f0a927ecc62ea2a4bef716f4fb9adabc06747fa` |
| `dev_id.jsonl` | 4,162 | `858e4ceaf3d27319292519266931b8fc0cfe8a1e23baecf903241a3c57e44388` |
| `validation.jsonl` | 10,106 | `d414891bf974805a4b98d51921435491fad6706a20d9fae06c9d4fd4a9839831` |
| `test.jsonl` | 42,221 | `fbe0cfcae2b99f98e7a7478e0e3945a29e72ac250feceef7b8ab452b7b4aff58` |
| `benchmarks/boardgameqa.jsonl` | 15,000 | `2b621172f4e5abc27a70be3ebe09afab0eebeb41e2058051a5eb77d3785db28b` |
| `benchmarks/dynabench.jsonl` | 543 | `cac2d2d113872de37651c78570273dd01a4924e3ae35d79f1c67705a6122b1c8` |
| `benchmarks/supernat.jsonl` | 4,100 | `cd7a731e58cfde149c288b40bc9274cccc7773a35ff7154630c8ed8436371939` |

`protocol.yaml` holds the hashes and sizes of `train`, `validation`, and `test`. A test compares them with the manifest,
and `load_split` compares them with the file. The manifest holds the hashes of the other files.

## What each split holds

- **train**: the synthetic units of the train block, without the dev_id units. It is the only training set of all models.
  Compositions: the 25 train compositions. Depth of the original member: 0 to 3.
- **dev_id**: an in-distribution dev set. It holds 5.2% of the train-block records
  (1,374 of 26,573 units, because the share applies to units). It has the same templates, compositions, and vocabulary as train.
  It is not for model selection. Step 5 compares the accuracy on dev_id (seen rules) with the accuracy on validation (unseen rules).
  The choice of the units is a seeded hash of `unit_id` (`seed` and `dev_share` in the config). It does not depend on the file order.
  All records of a unit stay together, so a pair or a paraphrase of a train record is never in dev_id.
- **validation**: the synthetic units of the val block. All model selection uses it. Its template group, its vocabulary pools,
  and its compositions (c05, c08, c11, c23) do not occur in train. Depths 0 to 5.
- **test**: the synthetic units of the test block (compositions c18, c28, c31, template group tg4, vocabulary pools v10 and v11, depths 0 to 5),
  and three held-out ProofWriter sets (open world, 3 classes, as the synthetic binary data):
  `depth5` (18,857 records), `birds_electricity` (5,270), and `pararules` (8,008).
  The build removes the 1,173 `depth5` records that the scan of the Ettin pretraining data flags
  (`contamination_removed` in the config, `configs/step4_contamination_removed.txt`, `docs/contamination.md`). Before this removal, `depth5` had 20,030 records.
  Step 8 reads this file once. Step 4 reads it only to build it, to hash it, and to run the leakage, id, and duplicate checks.
  The manifest holds the record counts of test (by source, `meta.subset`, and task type), its hash, and its leakage report.
  It holds no label count and no model statistic of test.
- **benchmarks** (evaluation only, never in training): `boardgameqa` (the official test file of each of the 15 subsets, `meta.subset` names the subset),
  `dynabench` (all 543 test examples), and `supernat` (41 held-out classification tasks, 100 instances each).

### ProofWriter test-only sets

The converter is `to_test_records` in `crux/data/proofwriter.py`. Only the split freezer calls it. It reads the
`meta-test.jsonl` of `OWA/depth-5`, `OWA/birds-electricity`, and `OWA/NatLang` (ParaRules) from the pinned ProofWriter zip.
The ids start with `pw-<subset>-`. `meta.subset` is `depth5`, `birds_electricity`, or `pararules`.
`meta.depth` is the depth of the question. `meta.dataset_depth` is 5 for `depth5`. The other two folders have no depth number,
so it is the `maxD` of the theory. The `depth5` set holds 10 questions of depth 6 or 7. They stay: the depth is in `meta.depth`.

All three files have the keys `triples`, `rules`, `theory`, and `questions` (with `answer` and `QDep`). The NatLang rows also have
`sentences` and `mappings`. Their `theory` is a paraphrase, so the plain `triples` and `rules` do not match it.
The check "facts + rules = theory" fails for all 482 rows. `split_theory` takes the fact sentences and the rule sentences from
`sentences` (the `mappings` link each triple and each rule to its sentence). One sentence can state several facts.
It needs the facts first and the rules after them. It compares the sentences with `theory` after it turns each ";" into "."
and merges the white space, because the release writes some sentence breaks as ";" in `sentences`. All 482 rows pass this check.
The question is the plain statement (for example "Alan is nice.") as in the other ProofWriter sets.

## Leakage check

The build runs the check before it records the hashes. Rules:

1. **Ids.** No two of these splits share a `template_id`, `template_group`, `composition_id`, `vocabulary_id`, `paraphrase_group`,
   `unit_id`, or `theory_hash`: train and validation, train and test, validation and test, dev_id and validation, dev_id and test.
   dev_id and train share no `unit_id`. The `pair_id` is not in the list, because it is `{unit_id}-{template_id}` (generator spec 4.5), so the `unit_id` check covers it.
   A shared id stops the build with an error. The generator makes the blocks, so a shared id is a bug and not a case to repair by removal.
2. **Exact duplicates.** No two files hold the same (`state`, `instruction`) text. The build removes the record from the later file of the order
   train, dev_id, validation, test, boardgameqa, dynabench, supernat.
3. **Near-duplicates.** The n-grams of a record are the word n-grams of `state + " " + instruction` (lowercase, tokens `[a-z0-9]+`). A record of an evaluation split is a
   near-duplicate if more than `max_overlap` of its n-grams occur in the reference split. Pairs: validation with train, test with train,
   each benchmark with train, and test with validation. dev_id is not checked against train, because it is in-distribution by design.
   The code hashes each n-gram to 64 bits and searches in a sorted `numpy` array, so memory use is small (about 8 bytes per distinct n-gram).

When the build removes a record, it removes the whole unit of a synthetic record, so the pairs stay complete. Then it checks again, until no record leaks.
The manifest lists the removed ids.

| Setting | Value | Reason |
|---|---|---|
| `ngram_n` | 8 | Choice of the author, no tuning. An 8-gram is long enough that short shared phrases rarely match, and short enough to catch a copied clause. |
| `max_overlap` | 0.10 | Choice of the author, no tuning. The check tolerates a small shared fragment. A record that shares more than 10% of its n-grams is a near-duplicate. |
| `dev_share` | 0.05 | Choice of the author. 1,374 units give 4,162 records, enough for a seen-rule accuracy with a narrow CI. The train split loses 5%. |
| `seed` | 20261016 | Seed of the `dev_id` hash. |
| `supernat_cap` | 100 | See below. |
| `boardgameqa_splits` | `[test]` | The official test file of each subset. The train and valid files are training material of the benchmark, and CRUX does not train on it. |

Overlap of the first round, as the share of n-grams in the reference split (lower is better). `Over` counts the records above `max_overlap`.

| Evaluation set ~ reference | Records | Max | p99 | Over |
|---|---|---|---|---|
| validation ~ train | 10,106 | 0.0000 | 0.0000 | 0 |
| test ~ train | 42,221 | 0.0000 | 0.0000 | 0 |
| boardgameqa ~ train | 15,000 | 0.0000 | 0.0000 | 0 |
| dynabench ~ train | 543 | 0.0000 | 0.0000 | 0 |
| supernat ~ train | 4,100 | 0.0000 | 0.0000 | 0 |
| test ~ validation | 42,221 | 0.0000 | 0.0000 | 0 |

The build removes no record, in any file. No id is shared. There is no exact duplicate. The maximum overlap is 0.0000 in all six pairs,
so the check does not depend on the exact value of `max_overlap`. The unit tests in `tests/test_splits.py` check that the
code flags a near-duplicate, removes its whole unit, and catches a shared id and an exact duplicate.

The check does not compare the test split with the public Ettin pretraining data. That is a separate check (Step 1, `docs/contamination.md`).

## Super-NaturalInstructions

- Recount of the Unknown-license tasks, from the pinned repository (`data/raw/supernat/.repo`).
  The command `python -m crux.splits sni-recount` makes it. The build does not run it (it needs the git repository).
  The result is in the manifest under `supernat`: of the 119 held-out tasks,
  47 are classification tasks with a fixed label set of 2 to 8 labels.
  6 of these 47 have the instance license "Unknown". `select_task` excludes them, which leaves 41 tasks.
  The Step 0 count of 16 Unknown-license tasks is for all 119 held-out tasks, not only the classification subset.
- Cap: the first 100 instances of each task, in file order. The official evaluation uses 100 instances per held-out task
  (Wang et al. 2022). The official loader of Tk-Instruct (`src/ni_dataset.py`) first takes the first 100 instances of a test task and then caps them at
  `max_num_instances_per_eval_task` = 100. Thus the first 100 in file order are the official instances. A seeded draw would not match them.
  Every task has at least 100 instances: 4,100 records = 41 tasks x 100 (142,350 before the cap).
  I read the loader on 2026-10-16 through a web fetch of the public repository. I did not run it.
- The file is for evaluation only. Do not redistribute the instances (`DATA_LICENSES.md`).

## Baselines

These are reference values, not model results. They are in `configs/splits_manifest.json` under `baselines`.

**Majority-class baseline per split and per source (accuracy, higher is better).** One definition: `crux.metrics.majority_baseline(records, ref=train)`.
The predictor outputs the most frequent label of the train records of the same task type. The label is the option text (binary) or the
target position (multi-choice). Only train labels give the prediction. The benchmarks have other label sets than train, so their
majority comes from their own records. A model must score clearly above these values.

| Split | Source | Records | Majority accuracy |
|---|---|---|---|
| train | synthetic/binary | 52,900 | 0.3551 |
| train | synthetic/multi_choice | 22,746 | 0.2946 |
| dev_id | synthetic/binary | 2,924 | 0.3824 |
| dev_id | synthetic/multi_choice | 1,238 | 0.2625 |
| validation | synthetic/binary | 7,086 | 0.3479 |
| validation | synthetic/multi_choice | 3,020 | 0.2983 |
| boardgameqa | boardgameqa | 15,000 | 0.3672 |
| dynabench | dynabench | 543 | 0.5083 |
| supernat | supernat | 4,100 | 0.4398 |

The protocol evaluates the test split once, in Step 8. Thus the manifest and this page hold no label count, no majority value, and no model statistic of test.
The test baselines (majority, fact-count) are computed in Step 8, with the same code.

The train block has more Unknown labels than 1/3 in the synthetic binary data (Step 3). Thus its majority accuracy is above 0.333.

**Majority-class baseline per composition (accuracy, higher is better).** Synthetic binary records. The original (orig) and the edited (edit) members of the pairs are separate.
The majority is the label that is most frequent among the train records of the same composition and pair role. A composition that train does not have
(validation) uses the majority of all train records of that pair role.
The last column is the share of the label Unknown among the edit records. dev_id is in the manifest only: it has 40 to 90 records per composition. There is no row for test.
The compositions c00, c01, c02, c04, c06, c08, and c09 have no True-False flip, so about half of their edit records are Unknown.

| Split | Composition | Orig records | Orig majority | Edit records | Edit majority | Edit Unknown share |
|---|---|---|---|---|---|---|
| train | c00 | 1490 | 0.352 | 1490 | 0.513 | 0.513 |
| train | c01 | 1465 | 0.347 | 1465 | 0.547 | 0.547 |
| train | c02 | 1024 | 0.352 | 1024 | 0.519 | 0.519 |
| train | c03 | 1081 | 0.356 | 1081 | 0.347 | 0.347 |
| train | c04 | 1048 | 0.358 | 1048 | 0.523 | 0.523 |
| train | c06 | 1044 | 0.360 | 1044 | 0.509 | 0.509 |
| train | c07 | 1038 | 0.348 | 1038 | 0.372 | 0.372 |
| train | c09 | 963 | 0.365 | 963 | 0.535 | 0.535 |
| train | c10 | 1102 | 0.360 | 1102 | 0.362 | 0.362 |
| train | c12 | 909 | 0.353 | 909 | 0.381 | 0.381 |
| train | c13 | 997 | 0.357 | 997 | 0.368 | 0.368 |
| train | c14 | 1011 | 0.370 | 1011 | 0.347 | 0.347 |
| train | c15 | 983 | 0.361 | 983 | 0.350 | 0.350 |
| train | c16 | 1023 | 0.359 | 1023 | 0.366 | 0.366 |
| train | c17 | 1070 | 0.352 | 1070 | 0.379 | 0.379 |
| train | c19 | 986 | 0.353 | 986 | 0.340 | 0.340 |
| train | c20 | 973 | 0.353 | 973 | 0.364 | 0.364 |
| train | c21 | 1041 | 0.348 | 1041 | 0.404 | 0.404 |
| train | c22 | 984 | 0.353 | 984 | 0.376 | 0.376 |
| train | c24 | 1016 | 0.349 | 1016 | 0.343 | 0.324 |
| train | c25 | 1051 | 0.348 | 1051 | 0.371 | 0.371 |
| train | c26 | 990 | 0.348 | 990 | 0.352 | 0.352 |
| train | c27 | 1094 | 0.367 | 1094 | 0.360 | 0.360 |
| train | c29 | 1018 | 0.355 | 1018 | 0.377 | 0.377 |
| train | c30 | 1049 | 0.352 | 1049 | 0.370 | 0.370 |
| validation | c05 | 918 | 0.367 | 918 | 0.367 | 0.367 |
| validation | c08 | 862 | 0.365 | 862 | 0.577 | 0.577 |
| validation | c11 | 879 | 0.364 | 879 | 0.356 | 0.356 |
| validation | c23 | 884 | 0.339 | 884 | 0.362 | 0.362 |

**Fact-count baseline (accuracy, higher is better).** The probe of `crux/gen/audit.py` (`qsplit`, `_balanced`, `gbm`): a gradient-boosting classifier on the number of true, false, and missing facts
of the queried entity on the query path. It trains on the orig binary records of depth 1 or more in train, with the same number of each label in each composition.
It scores the orig binary records of depth 1 or more of dev_id and validation, balanced the same way. Chance is 0.333.
It does not read the test split.
Random facts with the same label give 0.47 (Step 3), so a part of this signal comes from the logic and not from the generator.

| Split | Records | Accuracy |
|---|---|---|
| dev_id | 654 | 0.4541 |
| validation | 1,725 | 0.3890 |

**Edit pairs per split (counts).** One pair is one orig record with its edit record (`pair_id`). `Rule-edit pairs` have an edit type of
`exception_remove`, `priority_swap`, `rule_remove`, `condition_negate`, or `threshold_shift`. A relevant stay is a stay with `meta.edit_relevant` true.
The other stays have it false. This includes every depth-0 binary stay, which is never relevant by definition.

| Split | Task type | Pairs | Flips | Relevant stays | Other stays | Rule-edit pairs | Rule-edit units |
|---|---|---|---|---|---|---|---|
| train | binary | 26,450 | 15,733 | 851 | 9,866 | 15,457 | 10,334 |
| train | multi_choice | 11,373 | 6,931 | 834 | 3,608 | 875 | 596 |
| dev_id | binary | 1,462 | 891 | 42 | 529 | 894 | 587 |
| dev_id | multi_choice | 619 | 358 | 63 | 198 | 54 | 35 |
| validation | binary | 3,543 | 2,152 | 116 | 1,275 | 1,631 | 1,067 |
| validation | multi_choice | 1,510 | 880 | 111 | 519 | 44 | 31 |

The multi-choice rule-edit units of train and dev_id add up to the 631 of Step 3 (596 + 35). Validation has 31 units, all in c23.
Report the validation results for these pairs as one-composition results.

Only a small part of the stays is relevant: 1,009 of 12,679 binary stays and 1,008 of 5,333 multi-choice stays in train, dev_id, and validation together.
The generator selects stays by a looser test (any derived attribute of the queried entity changes), and the audit (`inert_stays`) checks that test.
Spec section 6 had a wrong "80% relevant" sentence. Step 4 corrected it (2026-10-16). In these splits, none of the 3,391 binary stays at depth 1 is relevant, and 393 of 3,499 at depth 2 are.
The paired accuracy on relevant stays has few pairs in validation (116 binary, 111 multi-choice). Report its CI.

## Freeze rule

The manifest is the freeze. `python -m crux.splits build` writes the files to `data/splits.new/` and compares their SHA-256 with the manifest.
It replaces the frozen files only if nothing changed or `--refreeze` is set. Otherwise it stops with an error and
leaves the frozen files and the manifest as they are. Use `--refreeze` only before Step 5.
Two builds on the same inputs give the same SHA-256 (checked).
The build reads `data/processed/synthetic/main` (Step 3), `data/processed/{boardgameqa,dynabench,supernat}` (Step 2), and the pinned ProofWriter zip.
