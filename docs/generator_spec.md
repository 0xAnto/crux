# Step 3 Generator Specification

This document is the contract for Step 3. Two parts implement it independently:

- **(a) Engine** `crux/gen/engine.py`: the executable symbolic rule engine. It validates a theory and computes the label and the depth of a query.
- **(b) Generator** `crux/gen/generator.py`: the deterministic generator. It builds theories, computes its own labels with its own code, renders the records, and asks the engine to check each record.

The engine and the generator must agree on every label, depth, and rule-type list. An example is rejected if they disagree. Both parts must pass the shared cases in `tests/data/engine_cases.json` (section 11).

Terms: a **theory** is the logical content (entities, attributes, facts, rules, exceptions, priorities). A **query** is the question about a theory. A **record** is one row of the schema in `docs/schema.md`. A **unit** is one original theory, its one counterfactual edit, one query, and 1 to 3 templates (section 5.1).

## 0. Files and owners

| File | Owner | Content |
|---|---|---|
| `crux/gen/__init__.py` | either | Empty. |
| `crux/gen/engine.py` | engine | Sections 1 to 3 and 10.1. No import from `crux.gen.generator`. |
| `tests/test_engine.py` | engine | Runs all shared cases (section 11). |
| `crux/gen/generator.py` | generator | Sections 4 to 9 and 10.2. Its own evaluator `solve`. |
| `crux/gen/templates.py`, `crux/gen/vocab.json` | generator | The 15 template families and the 12 vocabulary pools (section 4). |
| `configs/step3_generator.yaml` | generator | The values of section 9. |
| `tests/test_generator.py` | generator | Shared cases with `solve`, determinism, balance, vocabulary, and template checks. |
| `tests/data/engine_cases.json` | this spec | 51 shared cases. Do not change the expected values. Add cases at the end only. |
| `tests/data/denylist_tokens.txt` | this spec | Tokens that a vocabulary pool must not use (section 4.3). |

## 1. Logical representation

A theory is one JSON object with exactly these 6 keys. The order of each list has no effect on the semantics.

```json
{
  "entities":   ["e0", "e1"],
  "attributes": {"a0": "bool", "a1": "int", "a2": "bool"},
  "facts":      [{"entity": "e0", "attr": "a0", "value": true},
                 {"entity": "e0", "attr": "a1", "value": 7}],
  "rules":      [{"id": "r0", "if": [{"attr": "a0", "neg": false}, {"attr": "a1", "op": ">=", "value": 5}],
                  "then": {"attr": "a2", "neg": false}}],
  "exceptions": [{"id": "x0", "if": [{"attr": "a0", "neg": true}], "blocks": "r0"}],
  "priorities": [{"higher": "r1", "lower": "r0"}]
}
```

| Key | Form | Rules |
|---|---|---|
| `entities` | list of str | 1 or more. Each matches `e[0-9]+`. Unique. |
| `attributes` | dict str → `"bool"` or `"int"` | Each key matches `a[0-9]+`. |
| `facts` | list of `{entity, attr, value}` | The entity and the attribute exist. A `bool` attribute takes `true` or `false`. An `int` attribute takes an integer 0 to 99 (not a JSON bool). One fact at most for each (entity, attr). |
| `rules` | list of `{id, if, then}` | `id` matches `r[0-9]+`, unique. `if` is a list of 1 or more conditions with different attributes. `then` is a bool literal `{attr, neg}` on a `bool` attribute. |
| `exceptions` | list of `{id, if, blocks}` | `id` matches `x[0-9]+`, unique. `if` as for rules. `blocks` is the id of a rule in `rules`. |
| `priorities` | list of `{higher, lower}` | Two different rule ids. Both rules conclude the same attribute with opposite `neg`. No duplicate pair. The directed graph higher → lower has no cycle. |

A **condition** has one of two forms:

- Bool literal `{"attr": a, "neg": false|true}`. The attribute must be `bool`.
- Threshold `{"attr": a, "op": op, "value": k}`. The attribute must be `int`. `op` is one of `">="`, `">"`, `"<="`, `"<"`, `"=="`. `k` is an integer 0 to 99.

A **derived attribute** is an attribute that is the `then` attribute of 1 or more rules. All other attributes are **base attributes**. A fact must not give a value to a derived attribute. Thus facts give only base values, and rules give only derived values. `int` attributes are always base attributes.

There are no relations between entities. Every rule and every exception applies to each entity separately ("for each entity X: if X ... then X ..."). Thus the value of (e, a) depends only on the facts of e.

### 1.1 Query

```json
{"type": "binary", "entity": "e0", "attr": "a2", "neg": false}
{"type": "multi_choice", "entity": "e0", "candidates": ["a5", "a2", "a7"]}
```

- `binary`: the literal (entity, attr, neg). The attribute must be `bool`.
- `multi_choice`: "Which of these candidates holds for the entity?" `candidates` is a list of 2 to 8 different `bool` attributes. Each candidate is the positive literal (entity, candidate). The order of `candidates` is the order of the options in the record.

## 2. Semantics

### 2.1 Choices and reasons

- **Open world.** A value can be true, false, or unknown. A missing fact means "unknown", not "false". Reason: the label set needs "Unknown", as in ProofWriter OWA and BoardgameQA.
- **Classical (strong) negation, no negation as failure.** A negative literal holds only when a fact or a rule gives the value false. Reason: negation as failure makes "not a" hold whenever "a" is not derived. That removes the difference between False and Unknown.
- **Exceptions and priorities are the only non-monotonic parts.** An exception blocks one rule (an undercutting defeater: it does not assert the opposite). A priority decides between two firing rules with opposite conclusions (a rebutting defeat).
- **Stratified and acyclic.** The attribute dependency graph (section 2.3) must have no cycle. Thus one pass in topological order computes one unique model.
- **An unresolved conflict makes the theory invalid.** The label set has no "contradiction" label. If an unresolved conflict gave "Unknown", "Unknown" would have two meanings ("no information" and "contradiction"). The generator rejects such theories.

### 2.2 Values and conditions

`val(e, a)` is `true`, `false`, or `None` (unknown) for a bool attribute, and an integer or `None` for an int attribute. A base value comes from its fact, or is `None` when there is no fact.

A condition **holds** for entity e if and only if:

- Bool literal `{a, neg}`: `val(e, a)` is not `None` and `val(e, a) == (not neg)`.
- Threshold `{a, op, k}`: `val(e, a)` is not `None` and `val(e, a) op k` is true.

A condition on an unknown value never holds. This applies also to `neg: true` and to `a >= 0`.

### 2.3 Evaluation order

The **attribute dependency graph** has one node for each attribute and these edges:

- For each rule r and each condition c of r: `c.attr → r.then.attr`.
- For each exception x and each condition c of x: `c.attr → (rule x.blocks).then.attr`.

The graph must have no cycle (a self-loop is a cycle). Compute the derived attributes in any topological order. Each derived attribute is computed once, after all attributes that point to it.

### 2.4 Firing and conflicts

For a derived attribute a and an entity e:

1. A rule r with `r.then.attr == a` is **blocked** for e if an exception x with `x.blocks == r.id` exists and all conditions of x hold for e.
2. r **fires** for e if all conditions of r hold for e and r is not blocked for e.
3. `POS` = the firing rules with `then.neg == false`. `NEG` = the firing rules with `then.neg == true`.
4. r **beats** s if and only if the pair `{"higher": r, "lower": s}` is in `priorities`. There is no transitive closure: only the listed pairs count.
5. Resolution:
   - `POS` and `NEG` are empty: `val(e, a) = None`.
   - Only `POS` is not empty: `val(e, a) = true`, and `WIN = POS`.
   - Only `NEG` is not empty: `val(e, a) = false`, and `WIN = NEG`.
   - Both are not empty: `WIN_POS` = the rules in `POS` that beat every rule in `NEG`. `WIN_NEG` = the rules in `NEG` that beat every rule in `POS`. If `WIN_POS` is not empty, `val(e, a) = true` and `WIN = WIN_POS`. If `WIN_NEG` is not empty, `val(e, a) = false` and `WIN = WIN_NEG`. If both are empty, this is an **unresolved conflict**: the theory is invalid.

`WIN_POS` and `WIN_NEG` cannot both be non-empty, because the priority graph has no cycle. There is no team defeat: one single rule must beat all firing opponents.

The engine computes the full model: every derived attribute for every entity. An unresolved conflict anywhere in the model makes the theory invalid, also when the query does not need that value (case `err_conflict_on_other_entity`).

### 2.5 Labels

Binary query (e, a, neg):

| `val(e, a)` | Label |
|---|---|
| `None` | `"Unknown"` |
| `true` and `neg == false`, or `false` and `neg == true` | `"True"` |
| `false` and `neg == false`, or `true` and `neg == true` | `"False"` |

Multi-choice query (e, candidates): a candidate c is **true** if `val(e, c) == true`. Exactly one candidate must be true. The other candidates can be false or unknown. The label is the id of the true candidate (for example `"a5"`). If 0 or 2 or more candidates are true, the query is invalid (error code `mc_answer`).

### 2.6 Validation errors

`validate(theory, query)` returns a list of strings. Each string starts with an error code, a colon, and a space, then free text (for example `"cycle: a1 -> a1"`). An empty list means valid. The shared cases check only the code.

| Code | Condition |
|---|---|
| `schema` | Wrong keys, wrong JSON types, wrong id pattern, duplicate id, duplicate fact, an int value out of 0 to 99, an empty `if`, two conditions on the same attribute in one `if`, a bad `op`. |
| `ref` | An unknown entity, attribute, or rule id (in facts, conditions, `blocks`, priorities). |
| `type` | A bool literal on an int attribute, a threshold on a bool attribute, a `then` on an int attribute, a fact value of the wrong type. |
| `derived_fact` | A fact on a derived attribute. |
| `priority` | A pair with the same rule twice, a pair whose rules do not conclude the same attribute with opposite `neg`, a duplicate pair, or a cycle in the priority graph. |
| `cycle` | A cycle in the attribute dependency graph. |
| `conflict` | An unresolved conflict for some entity and attribute. |
| `query` | A bad query: wrong keys or type, unknown entity, unknown or int attribute, fewer than 2 or more than 8 candidates, duplicate candidates. |
| `mc_answer` | A multi-choice query with 0 or 2 or more true candidates. |

`validate` runs the static checks (all codes except `conflict` and `mc_answer`) first. It computes the model, and does the `conflict` and `mc_answer` checks, only if there is no static error.

## 3. Depth

### 3.1 Value depth

`d(e, a)` is defined for each known value:

- A base value from a fact: `d = 0`.
- A derived value: `d(e, a) = min over r in WIN of rdepth(e, r)`, with `rdepth(e, r) = 1 + max over conditions c of r of cdepth(e, c)`. `cdepth` is 0 for a threshold condition and `d(e, c.attr)` for a bool literal (the literal holds, so its value is known).

`WIN` is the set from section 2.4. In a resolved conflict, only the rules that beat every firing opponent count (case `depth_counts_only_winning_rules`). Exceptions add no depth.

### 3.2 Attribute height

`h(a) = 0` if no rule concludes a. Otherwise `h(a) = max over rules r with r.then.attr == a of (1 + max over conditions c of r of h(c.attr))`. Exception conditions do not count. `h` is finite because the graph has no cycle.

### 3.3 Query depth

- Binary, label True or False: `depth = d(e, a)`.
- Binary, label Unknown: `depth = h(a)`.
- Multi-choice: `depth = d(e, c*)`, with c* the true candidate.

Depth 0 occurs only for base attributes. Step 4 uses `unit_depth` (the depth of the original member, section 5.3) to hold out by depth.

## 4. IDs, templates, and vocabulary

### 4.1 Rule-type features and composition_id

`features(theory)` returns the sorted list (in this fixed order) of the present features:

| Bit | Feature | Present if |
|---|---|---|
| 1 | `conj` | A rule or an exception has 2 or more conditions. |
| 2 | `neg` | A condition (in a rule or an exception) has `neg: true`. A negative conclusion or a false fact does not count. |
| 4 | `thr` | A condition (in a rule or an exception) is a threshold. |
| 8 | `exc` | `exceptions` is not empty. |
| 16 | `pri` | `priorities` is not empty. |

`composition_id = "c%02d" % (sum of the bits)`, from `c00` to `c31`. Multi-step chains are not a feature: the depth (section 3) measures them.

The `composition_id` of a unit is the value for its original theory. Both members of the unit carry it. The `rule_types` meta field holds `features` of the record's own theory. An `exception_remove` edit can remove `exc` from `rule_types` of the edited member.

### 4.2 Hold-out partition and blocks

The frozen protocol (`configs/protocol.yaml`, `splits.train`) says: a validation or test example can use a template ID, a composition ID, a vocabulary ID, or a paraphrase only if the train split does not contain it. If each ID were independent and uniform, only about 27% of the units would have only train IDs. Thus the generator samples a **block** for each unit first, and then draws all IDs of the unit from the partition of that block.

| Axis | train block | val block | test block |
|---|---|---|---|
| Template group (`template_group`) | `tg0`, `tg1`, `tg2` (`t00`–`t08`) | `tg3` (`t09`–`t11`) | `tg4` (`t12`–`t14`) |
| Vocabulary pool (`vocabulary_id`) | `v00`–`v07` | `v08`, `v09` | `v10`, `v11` |
| Composition (`composition_id`) | the other 25 (table below) | `c08`, `c05`, `c11`, `c23` | `c18`, `c28`, `c31` |
| Depth of the original member | 0 to 3 | 0 to 5 | 0 to 5 |
| Share of units | 0.80 | 0.10 | 0.10 |

**Stratified composition hold-out.** The spec fixes these 7 held-out compositions (7 of 32, 22%). The generator does not draw them at runtime.

| Features k | All with k | Held out | val | test | train |
|---|---|---|---|---|---|
| 0 | 1 | 0 | — | — | `c00` |
| 1 | 5 | 1 | `c08` (exc) | — | `c01`, `c02`, `c04`, `c16` |
| 2 | 10 | 2 | `c05` (conj, thr) | `c18` (neg, pri) | 8 others |
| 3 | 10 | 2 | `c11` (conj, neg, exc) | `c28` (thr, exc, pri) | 8 others |
| 4 | 5 | 1 | `c23` (conj, neg, thr, pri) | — | `c15`, `c27`, `c29`, `c30` |
| 5 | 1 | 1 | — | `c31` (all) | — |

Rules of the choice:

- About 20% of the compositions of each feature count k = 1 to 5 are held out, at least 1 for each count. k = 0 has only `c00`, so it stays in train.
- k = 2 and k = 3 have 2 held-out compositions each: one goes to val and one to test. k = 1, 4, and 5 have only 1 each, so they cannot go to both splits. They alternate: k = 1 and k = 4 go to val, k = 5 goes to test. Thus val has the counts 1, 2, 3, 4, and test has the counts 2, 3, 5.
- Each rule type occurs in exactly 4 held-out compositions, and in val and in test.
- Train coverage: each rule type occurs in 12 of the 25 train compositions (0.48). All 10 pairs of rule types occur in train. 8 pairs have their own 2-feature train composition. conj + thr occurs in `c07` and neg + pri occurs in `c19`.

Depths 4 and 5 occur only in validation and test (as ProofWriter: train on depth 3 or less).

**Reporting.** Step 4 and Step 8 report unseen-rule accuracy overall, and also for each feature count of the unit's `composition_id`. Thus complexity is a separate breakdown and does not hide inside the overall value.

Step 4 uses `block` as the split and checks the leakage rules. Step 4 must keep all records of a unit in one split.

### 4.3 Vocabulary pools (vocabulary_id)

There are 12 pools, `v00` to `v11`, in `crux/gen/vocab.json`. Each pool is one domain of operational decisions (for example lending, parcel delivery, cloud hosting, clinic scheduling, research funding, software release, hiring, insurance, rentals). The generator agent writes the pools. Each pool has:

- `kind`: one singular noun for the entities (for example `"applicant"`).
- 12 or more entity names.
- 24 or more bool attributes. Each has a positive and a negative predicate phrase (for example "is verified" / "is not verified").
- 4 or more int attributes. Each has a fact frame and a threshold frame. Use measure phrases that need no plural agreement (for example "has a risk level of 7").
- One fact frame for each form: bool positive, bool negative, int. The state uses these frames only (section 5.2).

Requirements (`tests/test_generator.py` checks them):

1. The **content tokens** of a pool are the lowercase tokens (`[a-z]+`) of all its strings, minus the STOP tokens: `a an the is are not no has have does do of to for with in on at by from and or its it be been as per`.
2. The content tokens of two different pools are disjoint.
3. No content token of a pool is in `tests/data/denylist_tokens.txt`. This file holds the tokens of BoardgameQA (train split, all subsets), RuleTaker and ProofWriter (train split), and a manual list for Birds-Electricity and ParaRules. Thus the synthetic text does not copy their vocabulary.
4. Inside one pool, all names and phrases are different.

A unit's **lexicon** maps each entity id to a different entity name and each attribute id to a different attribute of the same type, from one pool. The lexicon is the same for all records of the unit.

### 4.4 Template families (template_id)

There are 15 template families, `t00` to `t14`, in 5 template groups of 3: `template_group = "tg%d" % (n // 3)` for `t{n:02d}`. A template family renders only the instruction (section 5.2). It gives these frames:

- the rule label for the rule at position i (for example "(1)", "Clause 1", "Item A"),
- a rule with 1 to 3 conditions and one conclusion,
- a bool condition (positive, negative) and a threshold condition (one phrase for each of the 5 operators),
- a conclusion (positive, negative),
- an exception ("label of the blocked rule" + conditions),
- a priority ("label of the higher rule" + "label of the lower rule"),
- the binary question (positive literal, negative literal) and the multi-choice question.

Requirements:

1. No two families use the same frame string for the same slot. Each family uses its own connectives and its own rule label style.
2. No frame contains one of these BoardgameQA or ProofWriter phrases or tokens (case-insensitive): a rule label of the form `Rule<number>`, "rule", "rules", "preferred", "preference", "conclude", "conclusion", "regarding", "game", "player", "at least one", "be certain", "positive that", "in the case where", "whenever", "surely", "unquestionably", "for sure", "careful", "problematic", "belief", "something", "someone", "things", "people".
3. No frame contains the text "Question:". The renderer adds it once (section 5.2).
4. Every family can render every theory and every query of section 1.

All families of a group are **paraphrase templates** for the units of that group. Families of the held-out groups (`tg3`, `tg4`) are the **held-out templates**. A paraphrase group never mixes template groups.

### 4.5 Unit, pair, and paraphrase ids

| Id | Format | Meaning |
|---|---|---|
| `unit_id` | `"u%06d" % k` | Unit number k, from 0. A unit that fails all attempts is skipped (section 9.3). Its number is not used again. |
| record `id` | `"syn-{unit_id}-{template_id}-{pair_role}"` | Unique in the file. |
| `pair_id` | `"{unit_id}-{template_id}"` | The original and the edited record with the same template. |
| `paraphrase_group` | `"{unit_id}-{pair_role}"` | The records with the same theory, query, and lexicon, and different templates. |

## 5. Records

### 5.1 Unit structure

A unit has: a block, a template group, an ordered list of 1 to 3 different templates of that group, a vocabulary pool, a lexicon, one query, the original theory T, the edited theory T' (T with exactly one edit, section 6), and the edit. Both members use the same query and the same lexicon.

For each template in the list, the unit gives 2 records: `orig` (T) and `edit` (T'). Thus a unit gives 2, 4, or 6 records.

### 5.2 Rendering

Rendering is a pure function of (theory, query, lexicon, template_id). It agrees with the existing converters:

- `state` = `" ".join(fact sentences)`, one sentence for each fact, in the order of `theory["facts"]`. The fact frames come from the vocabulary pool, not from the template. Thus all paraphrases of a unit member have the same `state`, and architecture D can reuse one cached state encoding.
- `instruction` = `" ".join(rule sentences + exception sentences + priority sentences + ["Question: " + question])`. Rules come in the order of `theory["rules"]`, then exceptions in their order, then priorities in their order. The rule at position i (0-based) has the label number i + 1. Exceptions and priorities refer to rules by these labels. `"Question:"` occurs exactly once, at the start of the last part (the batching code splits there).
- `options`: binary `["True", "False", "Unknown"]` (the canonical open-world order of `docs/schema.md`). Multi-choice: the positive predicate phrase of each candidate, in the order of `query["candidates"]`.
- `target`: binary `options.index(label)`. Multi-choice `query["candidates"].index(label)`.
- Every sentence ends with "." and the question ends with "?". Numbers are digits.

Two different theories must give two different texts (the lexicon is injective and the frames are unambiguous). Thus the two members of a pair always differ in `state` or in `instruction`.

Length caps (word count by `str.split()`): `state` 120 words or fewer, `instruction` 260 words or fewer. A test tokenizes a sample of 2,000 records with the tokenizer of `configs/step2_data.yaml`. No record may be truncated with those limits. If records are truncated, lower the caps. Reason for 260: a multi-choice option needs its own rules (about 16 words for each rule), so 150 words allowed only 2 or 3 options at depth 3 and more. For this cap, Step 2 sets `max_instruction_length` to 448 tokens and `max_length` to 640 tokens. The word cap also limits the option count of a deep unit (`max_options` in the config, for each depth).

## 6. Counterfactual pairs

The edited theory T' is T with exactly one change. All other list items keep their positions.

| `edit_type` | Change | `edit_path` | Needs |
|---|---|---|---|
| `fact_flip` | A bool fact value becomes its opposite. | `/facts/i/value` | — |
| `fact_remove` | One fact is deleted. T' keeps 1 fact or more. | `/facts/i` | — |
| `fact_number` | An int fact value v becomes v' ≠ v, 0 to 99. | `/facts/i/value` | `thr` |
| `threshold_shift` | A threshold value k (in a rule or an exception) becomes k' ≠ k, 0 to 99. | `/rules/i/if/j/value` or `/exceptions/i/if/j/value` | `thr` |
| `exception_remove` | One exception is deleted. | `/exceptions/i` | `exc` |
| `priority_swap` | `higher` and `lower` of one pair change places. | `/priorities/i` | `pri` |
| `rule_remove` | One rule is deleted. | `/rules/i` | — |
| `condition_negate` | One condition of one rule becomes its negation: a bool literal changes `neg`; a threshold op becomes its complement (`>=` and `<`, `>` and `<=`). A `==` threshold has no complement in the op set and is not a candidate. | `/rules/i/if/j/neg` or `/rules/i/if/j/op` | — |

The first three types change the state (state side). The other five change the rules (rule side). `rule_remove` and `condition_negate` are permitted in every composition, because every theory has rules.

Rules for a pair:

- T' must be valid, and the query must be valid for T' (for multi-choice: exactly one true candidate). If an edit makes T' invalid (for example `rule_remove` of a rule that an exception or a priority names, or `condition_negate` that causes an unresolved conflict), the generator skips that candidate.
- T' can have a feature that T does not have (`condition_negate` can add `neg`). `rule_types` shows the features of each member.
- `edit_relevant` is true if the labels differ, or if the edit changes the value of a derived attribute of the queried entity on the path of the query. The path holds the query attributes (the candidates for multi-choice) and all derived attributes that they depend on through rule conditions and exception conditions. The generator gives a stay a relevant edit when it can, for 80% of the stays or more. A depth-0 binary stay is never relevant, because its query attribute has no rules.
- The engine computes the labels of both members. `label_flip` is true if the labels differ. For multi-choice, a flip means that the true candidate changes.
- Because rules apply to each entity separately, an edit of a fact changes only the values of that fact's entity. Thus a flip by a fact edit must change the queried entity. This is why multi-choice candidates are attributes of one entity, not entities.
- The two members of a pair have the same `template_id`, `composition_id`, `vocabulary_id`, `block`, `edit_type`, `edit_path`, `label_flip`, `edit_relevant`, `pair_id`, and `unit_depth`. Their `depth` and `rule_types` can differ.

## 7. Paraphrases

A paraphrase renders the same (theory, query, lexicon) with another template family of the same template group. Only the instruction text changes. The theory, the query, the state, the options, the target, and the label stay the same. Logical equivalence is guaranteed by construction: all logical content comes from the theory and the query, and a template changes only the words around it.

`generate` checks each paraphrase group: the same `theory_hash`, `query`, `state`, `options`, and `target`, and all `instruction` texts different.

The number of templates in a unit is 1, 2, or 3, with probabilities 0.6, 0.3, and 0.1. The templates are a sample without repetition from the unit's template group.

## 8. Determinism

One base seed (`seed` in `configs/step3_generator.yaml`, default `20261015`) controls everything. The same seed, the same config, and the same Python minor version give byte-identical `all.jsonl`, `theories.jsonl`, and `report.json`.

```python
def derive_seed(base_seed: int, *parts) -> int:
    key = "/".join(str(p) for p in (base_seed, *parts)).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")

plan_rng    = random.Random(derive_seed(seed, "plan", k))        # plan of unit k (section 9.2)
attempt_rng = random.Random(derive_seed(seed, "attempt", k, j))  # attempt j of unit k
```

Rules:

- Use only `random.Random` objects made with `derive_seed`. Do not use the module-level `random` functions, `numpy.random`, `hash()`, the time, or the environment.
- Do not iterate over a `set` or a `frozenset` without `sorted()` (string hashes change between processes).
- Units run in order k = 0, 1, 2, .... The duplicate check (section 9.3) depends on the earlier units, so the run is sequential.
- `report.json` holds no time stamp and no path that depends on the machine.

## 9. Size, balance, and rejection

### 9.1 Size

`total_records` = 100,000, both members of every pair and all paraphrases included. Each unit adds an even number of records. When the remaining count r is less than the size of the next unit, that unit keeps only its first r / 2 templates. Expected: about 33,300 units and 66,700 different (theory, query) pairs.

### 9.2 Plan and targets

`plan_rng` draws the plan of unit k in this order: block, template group and templates, vocabulary pool, composition, task type, depth, option count, label target, flip or stay, edit type, query `neg`, edit label. The plan flip probability of a unit is `flip_share` + `flip_gain` × (the flips that its composition misses so far), clipped to 0 to 1, so that each composition stays near `flip_share`.

The construction does not use the flip status, the edit type, or the edit label (section 9.5). After the build, `choose_edit` solves all candidate edits of all permitted types. It keeps a flip, or a stay that changes a derived value of the queried entity (for binary depth 0, any stay). If an edit type has no such stay, a stay that changes no derived value (an inert stay) can take its place while the inert stays are at most `inert_share` of the stays. This cap does not change during the attempts of one unit, so a unit without a relevant stay failed all its attempts (multi-choice units with `conj` in c05, c09, c17, and c25 filled only 0.87 to 0.95 of their cells). Thus the second half of the attempts of a unit permits an inert stay. The key of an edit is (task type, edit type, signature). The signature is the change of shallow counts of the theory: facts, true and false facts, rules, exceptions, priorities, conclusion polarities, read counts, literal kinds, heights, the rules, exceptions, and priorities on the query attributes, and the true, false, and missing bool facts and the int facts of the queried entity. For each key, the generator counts the accepted flips F and stays S. With r = `flip_share` / (1 − `flip_share`), the room of a flip is r·S − F, and the room of a stay is F/r − S. Each edit also has a coarse key (task type, edit type, and only the changes of facts, true and false facts, rules, exceptions, priorities, heads with one conclusion polarity, the maximum height, the counts on the query attributes, and the facts of the queried entity). The room of an edit is the smaller of the rooms of its two keys. The room is a preference, not a limit. An earlier version closed a key at a room of −1 or less. Flips to one label need few specific edits, so their keys closed, and the stays that open a key again seldom arrived: one unit in eleven failed all 100 attempts. The generator takes the planned edit type if it has a flip (for a binary unit, a flip to the edit label) and a stay. The quota has no effect on this test. Else it takes the first other permitted type that has both, the type with the fewest accepted units first (ties in a random order), so that rare edit types reach their share. This test is the same for a flip plan and a stay plan, so the theories and the edit types of flips and stays come from the same selection, and the flip destinations of binary units follow the edit label. Then it takes the edit of the planned flip status with the most room, so that each key moves to flips : stays = r : 1 and no shallow count of the pair shows the flip status. At binary depth 0, a stay must change a fact of the queried entity, because every depth-0 flip does (a stay on another entity would show the stay), and no inert stay is permitted. A binary depth-0 Unknown has no flip (no edit adds a fact), so it takes the `fact_flip` or `fact_remove` stay with the most room, without the type test. These are the edit types of the depth-0 flips: a rule edit at depth 0 is always a stay. An earlier version used the type test for these units too. Then the rarest type with a stay took most of their stays: 40% of the `exception_remove` units and 28% of the `priority_swap` units were such stays, and the flip shares of these types fell to 0.36 and 0.40. The edit type then showed the flip status. A True↔False flip needs `pri`, or a twin: P and N of a binary unit read complementary literals of one base attribute. With `neg` or `thr` and with `conj` or `exc`, almost every binary theory has a twin (not if the twin falls on the attribute of the `conj` trunk, at depth 1). Without `conj` and `exc`, no theory has one: with one condition per rule, a theory with a twin is Unknown only for a missing fact, and no edit adds a fact, so an Unknown unit with a twin has no flip. A twin in some theories only would then show the label, because the accepted Unknown units would have none. Thus in c00, c01, c02, c04, c06, c08, and c09 no edit turns True into False at depth 1 or more, and a True or False unit at depth 1 or more has the edit label Unknown. At depth 0, a `fact_flip` of the query fact turns True into False in all compositions. An earlier version gave the edit label Unknown to depth-0 units of these 7 compositions too, so all 622 of them were `fact_remove` units. An Unknown unit has no edit label: it can flip to True or False. If no edit type has both, the attempt fails with `no_balanced_edit`. The targets below apply to the output. The run fails (non-zero exit) if a target with a tolerance is missed.

| Quantity | Target | Tolerance |
|---|---|---|
| Block shares (units) | train 0.80, val 0.10, test 0.10 | ± 0.01 |
| `multi_choice` share (units, each block) | 0.30 | ± 0.02 |
| Option count of `multi_choice` (units) | uniform from 2 to `max_options` of the unit (composition with or without `conj`, depth); the target is the mean of these shares | ± 0.02 each |
| Binary labels (orig records at depth ≥ 1, each block) | True, False, Unknown: 1/3 each | ± 0.03 (train), ± 0.05 (val, test) |
| Binary labels (all records, each block) | True, False, Unknown: 1/3 each | ± 0.05 |
| Multi-choice target position (records, for each option count n) | uniform, 1/n each | ± 0.03 |
| Depth of `orig` (units, train block) | 0: 0.10, 1: 0.30, 2: 0.30, 3: 0.30 | ± 0.02 |
| Depth of `orig` (units, val and test blocks) | 0 to 5: 1/6 each | ± 0.04 |
| Composition (units) | uniform over the compositions of the block | each ≥ half its uniform share |
| Rule-type coverage (units, train block) | each of the 5 features in ≥ 0.40 of units (expected 0.48) | — |
| Label flips (pairs) | `label_flip` true in 0.60 | ± 0.05; ≥ 0.40 in each task type |
| Edit types (pairs) | the edit types that the composition permits; the weight of a type is (mean accepted units of a type) / (accepted units of the type), at most 4 | each type ≥ 0.05 of pairs (`exception_remove`, `priority_swap`: ≥ 0.005); each type has ≥ 0.10 flips and ≥ 0.10 stays |
| Templates per unit | 1, 2, 3 with 0.6, 0.3, 0.1 | ± 0.02 |
| Binary query `neg` true (units) | 0.25 | — |

Rare edit types: an `exception_remove` or `priority_swap` unit needs one theory with two such edits: one that flips the label (for a binary unit, to the edit label) and one that is a relevant stay. The structure gives this pair seldom, and the plan weight (at most 4) does not change this. The shares were 0.010 and 0.012 while depth-0 Unknown units could take any stay type. After the rule that gives these units only the stay with the most room (then only `fact_flip` and `fact_remove` stays), the shares fell to about 0.006 and 0.009, because about 40% and 28% of the units of these types were such stays. The target for these two types is thus ≥ 0.005, and an earlier version had ≥ 0.05. The per-block counts are small (Step 3 report), so per-type results for these two types are not reported for the val and test blocks. The flip share of each type stays in 0.10 to 0.90.

Binary label plan: a stay pair keeps the label of `orig`. A flip pair prefers the edit label (one of the 2 other labels, uniform). At depth 1 or more, the orig label weight is 1/3 + `label_gain` × (the deficit of that label in the binary orig records at depth 1 or more of the composition so far), at least 0.02. This keeps the orig labels of each composition near 1/3 each. The edit label of a True or False unit uses the same rule: the weight of each permitted edit label is 1/3 + `label_gain` × (the deficit of that label in the edit members of the binary flips at depth 1 or more of the composition so far), at least 0.02, so that the flips do not end in Unknown more often than in True or False. Both controllers count per composition. At depth 0, the orig labels have uniform weights, and a flip plan cannot take Unknown (no edit adds a fact). Thus about 0.14 of the depth-0 orig records are Unknown. A depth-0 flip ends in Unknown with weight 2/3 and in the other known label with weight 1/3, so that all depth-0 records (orig and edit) have about 1/3 of each label. An earlier version counted depth 0 in the orig controller. Then the deeper units of a composition had more Unknown labels (up to 0.42), and the depth showed the label. An earlier version counted per block. In the 7 compositions without True↔False flips, most flips end in Unknown, so the block controllers gave the other compositions fewer Unknown labels (orig and edit) to balance the totals. The counts show the composition, so the counts then showed the label (`label_cue.counts` 0.04 at 100K). With per-composition controllers, the edit members of these 7 compositions keep more Unknown labels (67% of their flips end in Unknown), and a depth-0 Unknown is always a stay. Thus all binary records have about 0.357 Unknown labels in the train block. A label-dependent flip probability or orig label could balance the totals, but then the orig label would show the flip status. Thus the tolerance ± 0.03 applies to the orig records at depth 1 or more, and all records have the tolerance ± 0.05. An earlier version applied ± 0.03 to all records. The answer position of a multi-choice unit uses the same rule: 1/n + `position_gain` × (the deficit of that position in the records with n options). A binary unit takes the depth-0 share of the multi-choice units of its composition (multi-choice units have depth 1 or more), so that the depth of all units follows `depth_weights`.

Theory size (config defaults, not engine rules): binary units have 1 to 4 entities; multi-choice units have 1 to 4 entities, and the query entity is one of them. A rule or an exception has 1 to 3 conditions. Distractor rules and facts are present in every theory with rules. Distractors use only the features of the unit's composition. The features of the composition should occur on the derivation path of the query when the depth permits it. Every binary unit with depth 1 or more has a rule on the query attribute with the opposite conclusion, so that the two polarities in the text do not show the label. Most multi-choice distractors are attributes that a rule concludes positively but that do not hold for the queried entity (a condition fails or is unknown, an exception blocks the rule, or a rule with a higher priority defeats it).

### 9.3 Rejection

For attempt j of unit k, `generate` does these checks in this order. The first failed check rejects the full attempt (all its records), and the counter of that reason increases by 1:

| Reason | Check |
|---|---|
| `plan_miss` | The generator's own label, depth, features, or flip status differs from the plan. |
| `no_balanced_edit` | No edit type has a flip (to the edit label) and a stay (section 9.2). |
| `too_long` | A record breaks a length cap (section 5.2). |
| `duplicate` | A record has the same (`state`, `instruction`) text as an earlier accepted record. |
| `invalid:<code>` | `engine.validate(T, query)` or `engine.validate(T', query)` is not empty. `<code>` is the code of its first error. |
| `disagree_label` | The engine label differs from the generator label (either member). |
| `disagree_depth` | The engine depth differs from the generator depth (either member). |
| `disagree_features` | `engine.features` differs from the generator's features (either theory). |

After `max_attempts` (default 200) failed attempts, the unit is skipped and `unit_failed` increases by 1. With 100 attempts, the fact walk (section 9.5) left c11 at depth 5 (binary) with a fill rate of 0.94.

Disagreements show a difference between the two implementations, not bad luck. If `disagree_*` together exceed 0.1% of all attempts, `main()` exits with an error. `report.json` keeps the first 5 disagreeing (theory, query) pairs for each reason, so that they can become new shared cases.

### 9.4 Report

`report.json` holds: the config, the seed, the Python version, the counts (units attempted and accepted, records), the rejection counts by reason, the realized value of every target in section 9.2, the counts by `composition_id`, `template_id`, `vocabulary_id`, `edit_type`, and depth, the mean option count and the mean effective option count of the multi-choice records (the options that a rule concludes positively or that are a true fact of the queried entity), the number of truncated records from the tokenizer test if it ran, the fill rate of each cell (accepted units / planned units for each composition, depth, and task type), and the SHA-256 of `all.jsonl` and `theories.jsonl`.

### 9.5 Construction

The build makes the rules first and the facts last. The rule structure depends on the composition, the task type, the depth, and the option count. It does not depend on the label, the flip status, or the edit type. Thus the rules do not show the label or the flip status.

- Chain: level k (1 to D − 1) has a main rule that concludes the value that level k + 1 reads. With `pri`, a level can also have an opposite rule with a base condition and a priority in a random direction. With `conj`, a rule can have extra conditions.
- Binary unit: a rule P concludes the query attribute positively and a rule N concludes it negatively. Each has its own chain, or (with `conj`) they share the top of one chain and each has its own base condition. With `exc`, P and N each get an exception. With `pri`, a priority links P and N.
- Multi-choice unit: the options form pairs. The two options of a pair read one shared base attribute with opposite effects: complementary literals (`neg`, `thr`), or an exception or a stronger opposite rule on one option (`exc`, `pri`). Thus one fact edit can move the answer from one option to its partner. A rule edit cannot do this: it changes one rule, and the pair then has zero or two true options. Thus, with `pri` and depth 2 or more, one pair of each unit reads a derived attribute d instead (a contested pair): one option reads d, and a rule "d → not partner" beats the own rule of the partner. d has one positive rule, or (with probability 0.5) also a negative rule and a priority in a random direction. Then `rule_remove`, `condition_negate`, and `threshold_shift` on a rule of d, and `priority_swap` on the priority of d, can move the answer. d has height 1, so the chain of the option is one level shorter. Only one pair is contested, because more pairs add attributes beyond the pool size (section 4.3). Without `pri`, a multi-choice pair can change its answer only by a fact edit. The held-out compositions with `pri` are c23 (validation) and c18, c28, c31 (test).
- Distractors: side rules read a level literal or a used base attribute, so that a stay can change a derived value. The first side rule reads a new base attribute, so that the queried entity has a free fact (see the fact search below). Other entities get noise facts.
- If a feature of the composition is not in the structure yet, the build adds a rule, an exception, or a priority with that feature.

Then a fact search makes the planned label hold for the queried entity. A binary structure at depth 1 or more is accepted only if the search succeeds for all three labels. Else a structure that permits one label only (for example, P and N read the same literal: only Unknown) shows the label. To make a rule fire, the search makes all conditions hold and each exception on the rule fail. To stop a rule with an exception x, it makes x active with probability 1/(m+1) (m: the conditions of x), and the conditions of the rule hold too. Else it makes one condition fail (a condition that already fails, if there is one), and the other conditions hold. Counted over the conditions of the rule and of the exception, a stopped rule thus has on average as many holding conditions as a firing rule. An earlier version let the other conditions hold with probability 0.7, so the true facts on the query path showed the label. A derived condition fails by the opposite value (probability 0.7, if a rule can conclude it) or by an unknown value. If that value clashes with an earlier choice, the search tries the other one. For multi-choice, the answer must be true and each other option must be not true (unknown or false). The other base attributes of the queried entity (the free ones, also the ones that no rule reads) take random values, independent of the label: 3/8 true, 3/8 false, and 1/4 no fact (bool), and 3/4 value and 1/4 no fact (int, with a random value 0 to 99). An earlier version moved the totals of the queried entity toward target shares with the free facts. The free facts then leaned against the fixed facts, so the facts off the query path showed the label, and the audit of all facts together did not see it. The audit (`crux/gen/audit.py`) now tests the facts on and off the query path separately (`label_cue.counts`, `label_cue.path`). After the search, a Metropolis-Hastings walk draws again the facts of the queried entity that a condition reads (4 sweeps). Each move draws a value with the default shares, and the move stays only if the answer and the depth stay. The proposal is the prior, so the facts of a built theory tend to the label-blind distribution for the answer and the depth. `choose_edit` then keeps only theories with a balanced edit, and this shifts the facts of the accepted units: an Unknown theory with many missing facts often has no flip, because no edit adds a fact. Thus Unknown orig records have fewer missing facts on the path than the reference (0.47 against 0.85 in a review test). This lowers the signal of the facts. Without the walk, the facts on the path depended on the label in a different way in each composition, because an active exception needs true facts. In c08 (`exc` only, val), Unknown records had more true facts on the path than True and False records. This is the reverse of most train compositions, so a fact-count probe trained on the train block scored 0.29 on the val block (chance 0.33). If the instruction is longer than the word budget, the build starts again with fewer optional parts (up to 3 times). The generator's own `solve` checks the label and the depth of the result.

## 10. Interface

### 10.1 Engine (`crux/gen/engine.py`)

```python
LABELS = ("True", "False", "Unknown")

def validate(theory: dict, query: dict | None = None) -> list[str]:
    """Return the errors of section 2.6 ("<code>: <text>"). An empty list means valid."""

def evaluate(theory: dict, query: dict) -> dict:
    """Return {"label": str, "depth": int}, and optionally "proof": list[str] (rule ids).
    Binary: label in LABELS. Multi-choice: label is the id of the true candidate.
    Raise ValueError (with the errors) if validate(theory, query) is not empty."""

def features(theory: dict) -> list[str]:
    """Section 4.1, in the order conj, neg, thr, exc, pri. Reads only the lists; needs no model."""

def composition_id(theory: dict) -> str:
    """'c%02d' of the feature bits."""
```

The engine does not change its arguments. It uses only the standard library.

### 10.2 Generator (`crux/gen/generator.py`)

```python
def generate(config: dict, seed: int, stats: dict | None = None) -> Iterator[tuple[dict, dict, dict]]:
    """Yield (theory, query, record) for each accepted record, in output order (section 5.4).
    Call the engine for each record (section 9.3). If stats is a dict, add the counters of section 9.3 to it."""

def solve(theory: dict, query: dict) -> dict:
    """The generator's own evaluator: {"label": str, "depth": int}. Raise ValueError for an invalid theory or query.
    Must not import or call crux.gen.engine."""

def features(theory: dict) -> list[str]:
    """The generator's own copy of section 4.1. Must not import crux.gen.engine."""

def main() -> None:
    """python -m crux.gen.generator [--config configs/step3_generator.yaml] [--seed N] [--total N] [--out DIR]
    Write all.jsonl, theories.jsonl, and report.json (section 5.4)."""
```

The generator imports `crux.gen.engine` only inside `generate`, for the checks. Its labels must come from its own code (`solve` or the construction).

## 11. Shared test cases

`tests/data/engine_cases.json` holds 51 cases: 37 valid cases and 14 error cases. They cover facts and the open world, conjunctions, negated conditions without negation as failure, negative conclusions, all 5 threshold operators at the boundary, a missing numeric value, chains (depth 3, shortest proof, maximum over conditions, height for Unknown), exceptions (active, unknown condition, derived condition), priorities (both directions, one side fires, exception on the higher rule, one rule beats all, depth from the winning rule only), one theory with all 5 features and its threshold-shift counterfactual, 3 multi-choice cases, and the errors `conflict` (no priority, no team defeat, no transitivity, on another entity), `cycle` (rules, exception), `derived_fact`, `priority` (same polarity, cycle), `mc_answer` (two true, none true), `query`, `type`, and `ref`.

Form of one case:

```json
{"id": "...", "desc": "...", "theory": {...}, "query": {...},
 "expect": {"label": "True", "depth": 1, "features": ["thr"]}}
{"id": "...", "desc": "...", "theory": {...}, "query": {...}, "expect": {"error": "conflict"}}
```

Required tests:

- Engine: for each valid case, `evaluate` returns the expected `label` and `depth`, `validate` returns `[]`, and `features` returns the expected list. For each error case, `validate(theory, query)` has an error that starts with `"<error>:"`, and `evaluate` raises `ValueError`.
- Generator: for each valid case, `solve` returns the expected `label` and `depth`, and `features` returns the expected list. For each error case, `solve` raises `ValueError`.
- Both: the result does not change when the lists `facts`, `rules`, `exceptions`, and `priorities` are reversed.
- Generator: two runs with `total_records: 600` and the same seed give the same SHA-256 for each output file. Another seed gives a different file.

When a full run finds a disagreement, add the case to the end of the file after a check by hand. Do not change existing expected values.
