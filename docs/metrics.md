# Metrics (Step 4)

Code: `crux/metrics.py`. Tests: `tests/test_metrics.py`. Input: a list of schema records and a parallel list
of probability vectors (one value per option, sum 1). The prediction is the argmax. For a `scoring` record,
the correct option is the argmax of `target`. `evaluate(records, probs, train=None, perms=None)` returns all
metrics that apply as a flat dict. It skips a metric if the meta fields are absent. Each count has the key
`<metric>/n`. The module has no bootstrap: Step 8 does it.

If the records have more than one `source` or `meta.subset`, `evaluate` gives no pooled value. Each key has the prefix
`<source>/<subset>/` (for example `proofwriter/depth5/accuracy`), because a pooled accuracy over mixed sources is not a result.

Slices of the paired metrics use `unit_depth`, not `depth`. The orig and the edit record of a pair can have different `depth`
(165 of the 5,053 validation pairs do), but they have the same `unit_depth`.

| Metric | Definition | Unit | Better |
|---|---|---|---|
| `accuracy` | Share of records with the correct predicted option. | record | higher |
| `macro_f1` | Mean F1 over the option texts as classes. `evaluate` uses the binary records only. | record | higher |
| `ece` | Expected calibration error on the maximum probability. 15 equal-width bins on (0, 1] (Guo et al. 2017). The protocol does not fix the bin count. | record | lower |
| `paired_accuracy` | A pair (`meta.pair_id`) is correct only if the orig and the edit record are both correct. An incomplete pair raises an error. | pair | higher |
| `paired_accuracy/flip`, `/stay` | The same, for pairs with `label_flip` true or false. | pair | higher |
| `paired_accuracy/relevant_stay` | The same, for stays with `edit_relevant` true. | pair | higher |
| `paired_accuracy/unit_depth=<d>` | The same, for one `meta.unit_depth`. | pair | higher |
| `paired_accuracy/edit_type=<t>/orig=<label>` | The same, for one edit type and one orig label (the label text for binary records, else the task type). Do not report `exception_remove` and `priority_swap` for val (Step 8). | pair | higher |
| `paraphrase_consistency` | Share of paraphrase groups (`meta.paraphrase_group`, 2 or more records) where all members predict the same option text. A group with one record is skipped. | group | higher |
| `paraphrase_consistent_correct` | Share of those groups where all members predict the same correct option. | group | higher |
| `option_order_invariance` | Share of records whose predicted option text is the same under all option permutations. Run the model on `permute_options(record, seed)` for each seed and on the identity. Binary records are permuted too. `permute_options` raises an error for a record with fewer than 2 options. The function raises an error if the permutations differ in the number of records or predictions. | record | higher |
| `unseen_rule_accuracy` | Accuracy on records whose `template_id`, `composition_id`, and `vocabulary_id` all differ from the train records. A record with a train id raises an error (`drop_seen=True` drops it). `evaluate` scores only the records that have all three ids, drops the records with a train id, and reports their number as `unseen_rule_accuracy/seen_dropped`. On an in-distribution set (dev_id) it gives no accuracy and only this count. | record | higher |
| `unseen_rule_accuracy/features=<k>`, `/depth=<d>` | The same, for the number of set bits of `composition_id` and for `meta.depth`. | record | higher |
| `majority_baseline` | Accuracy of the most frequent label: the option text (binary) or the target index (other types), counted per task type. The majority comes from the train records when given. | record | higher |
| `majority_baseline/composition_id=<group>` | The same for each group of `composition_id`. A group uses the majority of the train records of the same group. A group that the train records do not have uses the majority of all train records. | record | higher |

The majority baseline is a reference, not a model result. Report it with every accuracy.
