"""Step 3: the executable symbolic rule engine.

The contract is docs/generator_spec.md, sections 1 to 3 and 10.1. This module is independent of
crux/gen/generator.py: it does not import it, and it computes each label with its own code.

Check the generator output: python -m crux.gen.engine check <records.jsonl> [--theories <theories.jsonl>]
"""
import argparse
import hashlib
import json
import operator
import re
import sys
from collections import Counter
from functools import cache
from pathlib import Path

LABELS = ("True", "False", "Unknown")
OPS = {">=": operator.ge, ">": operator.gt, "<=": operator.le, "<": operator.lt, "==": operator.eq}
FEATURE_BITS = (("conj", 1), ("neg", 2), ("thr", 4), ("exc", 8), ("pri", 16))
THEORY_KEYS = {"entities", "attributes", "facts", "rules", "exceptions", "priorities"}
QUERY_KEYS = {"binary": {"type", "entity", "attr", "neg"}, "multi_choice": {"type", "entity", "candidates"}}


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _id_ok(v, prefix):
    return isinstance(v, str) and re.fullmatch(prefix + "[0-9]+", v) is not None


def _conditions_errors(where, conds):
    """Section 1: the form of an `if` list. Return a list of schema errors."""
    if not isinstance(conds, list) or not conds:
        return [f"schema: {where}: 'if' must be a list with 1 or more conditions"]
    errors = []
    for c in conds:
        if not isinstance(c, dict) or not isinstance(c.get("attr"), str):
            errors.append(f"schema: {where}: bad condition {c!r}")
        elif set(c) == {"attr", "neg"}:
            if not isinstance(c["neg"], bool):
                errors.append(f"schema: {where}: 'neg' must be a bool in {c!r}")
        elif set(c) == {"attr", "op", "value"}:
            if c["op"] not in OPS:
                errors.append(f"schema: {where}: bad op in {c!r}")
            if not _is_int(c["value"]) or not 0 <= c["value"] <= 99:
                errors.append(f"schema: {where}: the threshold value must be an integer 0 to 99 in {c!r}")
        else:
            errors.append(f"schema: {where}: bad condition keys {sorted(c)}")
    if not errors and len({c["attr"] for c in conds}) != len(conds):
        errors.append(f"schema: {where}: two conditions on the same attribute")
    return errors


def _schema_errors(t):
    """The `schema` checks of section 2.6. The later checks need a theory without schema errors."""
    if not isinstance(t, dict) or set(t) != THEORY_KEYS:
        return [f"schema: a theory must be a dict with the keys {sorted(THEORY_KEYS)}"]
    errors = []
    ents, attrs = t["entities"], t["attributes"]
    if not isinstance(ents, list) or not ents or not all(_id_ok(e, "e") for e in ents):
        errors.append("schema: 'entities' must be a list of 1 or more ids e<n>")
    elif len(set(ents)) != len(ents):
        errors.append("schema: duplicate entity")
    if not isinstance(attrs, dict) or not all(_id_ok(a, "a") and v in ("bool", "int") for a, v in attrs.items()):
        errors.append("schema: 'attributes' must map ids a<n> to 'bool' or 'int'")
    for key in ("facts", "rules", "exceptions", "priorities"):
        if not isinstance(t[key], list) or not all(isinstance(item, dict) for item in t[key]):
            errors.append(f"schema: '{key}' must be a list of dicts")
    if errors:
        return errors

    seen = set()
    for f in t["facts"]:
        if set(f) != {"entity", "attr", "value"} or not isinstance(f["entity"], str) or not isinstance(f["attr"], str):
            errors.append(f"schema: bad fact {f!r}")
            continue
        if _is_int(f["value"]) and not 0 <= f["value"] <= 99:
            errors.append(f"schema: int value out of 0 to 99 in {f!r}")
        if (f["entity"], f["attr"]) in seen:
            errors.append(f"schema: duplicate fact for ({f['entity']}, {f['attr']})")
        seen.add((f["entity"], f["attr"]))

    ids = []
    for r in t["rules"]:
        if set(r) != {"id", "if", "then"} or not _id_ok(r["id"], "r"):
            errors.append(f"schema: bad rule {r!r}")
            continue
        ids.append(r["id"])
        errors += _conditions_errors(r["id"], r["if"])
        then = r["then"]
        if not isinstance(then, dict) or set(then) != {"attr", "neg"} or not isinstance(then["attr"], str) \
                or not isinstance(then["neg"], bool):
            errors.append(f"schema: {r['id']}: 'then' must be a bool literal {{attr, neg}}")
    for x in t["exceptions"]:
        if set(x) != {"id", "if", "blocks"} or not _id_ok(x["id"], "x") or not isinstance(x["blocks"], str):
            errors.append(f"schema: bad exception {x!r}")
            continue
        ids.append(x["id"])
        errors += _conditions_errors(x["id"], x["if"])
    if len(set(ids)) != len(ids):
        errors.append("schema: duplicate rule or exception id")
    for p in t["priorities"]:
        if set(p) != {"higher", "lower"} or not isinstance(p["higher"], str) or not isinstance(p["lower"], str):
            errors.append(f"schema: bad priority {p!r}")
    return errors


def _toposort(graph):
    """Return (order, None) with each node before its successors, or (None, cycle) if the graph has a cycle."""
    order, done, path = [], set(), []

    def visit(n):
        if n in path:
            return path[path.index(n):] + [n]
        if n in done:
            return None
        path.append(n)
        for m in sorted(graph.get(n, ())):
            cycle = visit(m)
            if cycle:
                return cycle
        path.pop()
        done.add(n)
        order.append(n)
        return None

    for n in sorted(graph):
        cycle = visit(n)
        if cycle:
            return None, cycle
    return order[::-1], None


def _static_errors(t):
    """The `ref`, `type`, `derived_fact`, `priority`, and `cycle` checks of section 2.6.
    Return (errors, order): order is the attribute order for the evaluation (section 2.3)."""
    errors = []
    attrs = t["attributes"]
    rules = {r["id"]: r for r in t["rules"]}
    derived = {r["then"]["attr"] for r in t["rules"]}

    for f in t["facts"]:
        if f["entity"] not in t["entities"] or f["attr"] not in attrs:
            errors.append(f"ref: unknown entity or attribute in fact {f!r}")
        elif f["attr"] in derived:
            errors.append(f"derived_fact: {f['attr']} is derived, but a fact gives it a value")
        # spec: a value that is not a bool or an int (for example a string) can be a "wrong JSON type" (schema)
        # or "a fact value of the wrong type" (type). This engine uses `type` for every value that does not
        # match the attribute type, and `schema` only for an int out of 0 to 99.
        elif not (isinstance(f["value"], bool) if attrs[f["attr"]] == "bool" else _is_int(f["value"])):
            errors.append(f"type: the value of {f!r} is not of type {attrs[f['attr']]}")

    for item in t["rules"] + t["exceptions"]:
        for c in item["if"]:
            if c["attr"] not in attrs:
                errors.append(f"ref: {item['id']}: unknown attribute {c['attr']}")
            elif ("op" in c) != (attrs[c["attr"]] == "int"):
                errors.append(f"type: {item['id']}: the condition {c!r} does not match the type of {c['attr']}")
    for r in t["rules"]:
        a = r["then"]["attr"]
        if a not in attrs:
            errors.append(f"ref: {r['id']}: unknown attribute {a}")
        elif attrs[a] != "bool":
            errors.append(f"type: {r['id']}: 'then' is on the int attribute {a}")
    for x in t["exceptions"]:
        if x["blocks"] not in rules:
            errors.append(f"ref: {x['id']} blocks the unknown rule {x['blocks']}")

    beats = {}
    for p in t["priorities"]:
        hi, lo = p["higher"], p["lower"]
        if hi not in rules or lo not in rules:
            errors.append(f"ref: unknown rule in priority {p!r}")
        elif hi == lo:
            errors.append(f"priority: the same rule twice in {p!r}")
        elif rules[hi]["then"]["attr"] != rules[lo]["then"]["attr"] or \
                rules[hi]["then"]["neg"] == rules[lo]["then"]["neg"]:
            errors.append(f"priority: {hi} and {lo} do not conclude the same attribute with opposite neg")
        elif lo in beats.get(hi, ()):
            errors.append(f"priority: duplicate pair {p!r}")
        else:
            beats.setdefault(hi, set()).add(lo)
    _, cycle = _toposort(beats)
    if cycle:
        errors.append("priority: cycle " + " > ".join(cycle))

    graph = {a: set() for a in attrs}
    for r in t["rules"]:
        for c in r["if"]:
            graph.setdefault(c["attr"], set()).add(r["then"]["attr"])
    for x in t["exceptions"]:
        if x["blocks"] in rules:
            for c in x["if"]:
                graph.setdefault(c["attr"], set()).add(rules[x["blocks"]]["then"]["attr"])
    order, cycle = _toposort(graph)
    if cycle:
        errors.append("cycle: " + " -> ".join(cycle))
    return errors, order


def _query_errors(t, q):
    """The `query` checks of section 2.6."""
    keys = QUERY_KEYS.get(q.get("type")) if isinstance(q, dict) else None
    if keys is None or set(q) != keys:
        return [f"query: wrong keys or type in {q!r}"]
    errors = []
    if q["entity"] not in t["entities"]:
        errors.append(f"query: unknown entity {q['entity']!r}")
    if q["type"] == "binary":
        if not isinstance(q["neg"], bool):
            errors.append("query: 'neg' must be a bool")
        names = [q["attr"]]
    else:
        names = q["candidates"]
        if not isinstance(names, list) or not 2 <= len(names) <= 8:
            return errors + ["query: 'candidates' must be a list of 2 to 8 attributes"]
        if len(set(map(json.dumps, names))) != len(names):
            errors.append("query: duplicate candidates")
    for a in names:
        if not isinstance(a, str) or t["attributes"].get(a) != "bool":
            errors.append(f"query: {a!r} is not a bool attribute")
    return errors


def _holds(val, e, c):
    """Section 2.2. A condition on an unknown value never holds."""
    v = val.get((e, c["attr"]))
    if v is None:
        return False
    if "op" in c:
        return OPS[c["op"]](v, c["value"])
    return v == (not c["neg"])


def _model(t, order):
    """Sections 2.4 and 3.1: compute every derived value for every entity.
    Return (val, dep, errors): val[(e, a)] and dep[(e, a)] exist only for known values."""
    beats = {(p["higher"], p["lower"]) for p in t["priorities"]}
    val, dep, errors = {}, {}, []
    for f in t["facts"]:
        val[f["entity"], f["attr"]] = f["value"]
        dep[f["entity"], f["attr"]] = 0
    for e in t["entities"]:
        for a in order:
            fire = [r for r in t["rules"] if r["then"]["attr"] == a
                    and all(_holds(val, e, c) for c in r["if"])
                    and not any(x["blocks"] == r["id"] and all(_holds(val, e, c) for c in x["if"])
                                for x in t["exceptions"])]
            pos = [r for r in fire if not r["then"]["neg"]]
            neg = [r for r in fire if r["then"]["neg"]]
            if pos and neg:
                # Only the listed pairs count: there is no transitive closure and no team defeat.
                win_pos = [r for r in pos if all((r["id"], s["id"]) in beats for s in neg)]
                win_neg = [r for r in neg if all((r["id"], s["id"]) in beats for s in pos)]
                if not win_pos and not win_neg:
                    errors.append(f"conflict: {e}, {a}: {sorted(r['id'] for r in fire)} fire, and no rule wins")
                    continue
                pos, neg = win_pos, win_neg
            win = pos or neg
            if not win:
                continue
            val[e, a] = bool(pos)
            dep[e, a] = min(1 + max(0 if "op" in c else dep[e, c["attr"]] for c in r["if"]) for r in win)
    return val, dep, errors


def _check(theory, query):
    """Return (errors, val, dep). val and dep are None if the theory or the query is invalid."""
    errors = _schema_errors(theory)
    # spec: section 2.6 does not say if the query checks run when the theory has schema errors.
    # They need a sound entity list and attribute dict, so this engine skips them in that case.
    if errors:
        return errors, None, None
    # spec: section 2.6 does not fix the order of the static errors, but section 9.3 counts the code of the
    # first error. This engine reports: schema; then ref, derived_fact, type (facts, conditions, 'then',
    # 'blocks'); then priority; then cycle; then query.
    errors, order = _static_errors(theory)
    if query is not None:
        errors += _query_errors(theory, query)
    if errors:
        return errors, None, None
    val, dep, errors = _model(theory, order)
    if not errors and query is not None and query["type"] == "multi_choice":
        true = [c for c in query["candidates"] if val.get((query["entity"], c)) is True]
        if len(true) != 1:
            errors.append(f"mc_answer: {len(true)} true candidates {true}, not exactly 1")
    return errors, val, dep


def validate(theory: dict, query: dict | None = None) -> list[str]:
    """Return the errors of section 2.6 ("<code>: <text>"). An empty list means valid."""
    return _check(theory, query)[0]


def height(theory: dict, attr: str) -> int:
    """Section 3.2. Exception conditions do not count. The theory must be valid (no cycle)."""
    @cache
    def h(a):
        return max((1 + max(h(c["attr"]) for c in r["if"]) for r in theory["rules"] if r["then"]["attr"] == a),
                   default=0)
    return h(attr)


def evaluate(theory: dict, query: dict) -> dict:
    """Return {"label": str, "depth": int} (section 2.5 and section 3.3).
    Binary: label in LABELS. Multi-choice: label is the id of the true candidate.
    Raise ValueError (with the errors) if validate(theory, query) is not empty."""
    errors, val, dep = _check(theory, query)
    if errors:
        raise ValueError("; ".join(errors))
    e = query["entity"]
    if query["type"] == "multi_choice":
        c = next(c for c in query["candidates"] if val.get((e, c)) is True)
        return {"label": c, "depth": dep[e, c]}
    a = query["attr"]
    v = val.get((e, a))
    if v is None:
        return {"label": "Unknown", "depth": height(theory, a)}
    return {"label": "True" if v == (not query["neg"]) else "False", "depth": dep[e, a]}


def _bits(theory):
    conds = [c for item in theory["rules"] + theory["exceptions"] for c in item["if"]]
    return {
        "conj": any(len(item["if"]) >= 2 for item in theory["rules"] + theory["exceptions"]),
        "neg": any(c.get("neg") is True for c in conds),
        "thr": any("op" in c for c in conds),
        "exc": bool(theory["exceptions"]),
        "pri": bool(theory["priorities"]),
    }


def features(theory: dict) -> list[str]:
    """Section 4.1, in the order conj, neg, thr, exc, pri. Reads only the lists; needs no model."""
    present = _bits(theory)
    return [name for name, _ in FEATURE_BITS if present[name]]


def composition_id(theory: dict) -> str:
    """'c%02d' of the feature bits."""
    present = _bits(theory)
    return "c%02d" % sum(bit for name, bit in FEATURE_BITS if present[name])


def theory_hash(theory: dict) -> str:
    """Section 5.4."""
    canonical = json.dumps(theory, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def check_record(record: dict, theory: dict | None) -> str:
    """Derive the label, the depth, and the rule types of one record again from its theory and query.
    Return "agree", or the first reason of the disagreement."""
    if theory is None:
        return "missing_theory"
    meta = record["meta"]
    query = meta["query"]
    errors = validate(theory, query)  # for multi_choice, this checks that exactly 1 candidate is true
    if errors:
        return "invalid:" + errors[0].split(":")[0]
    result = evaluate(theory, query)
    options = query["candidates"] if query["type"] == "multi_choice" else record["options"]
    if options[record["target"]] != result["label"]:
        return "disagree_label"
    if meta["depth"] != result["depth"]:
        return "disagree_depth"
    if meta["rule_types"] != features(theory):
        return "disagree_features"
    # The composition_id of a unit is the value of its original theory (section 4.1).
    if meta["pair_role"] == "orig" and meta["composition_id"] != composition_id(theory):
        return "disagree_composition"
    return "agree"


def check_files(records_path, theories_path):
    """Check each record of a JSONL file against its theory. The record links to its theory by
    meta.theory_hash. The hash is computed again from each theory, so a wrong stored hash gives missing_theory.
    Return (counts by reason, up to 5 record ids for each reason that is not "agree")."""
    theories = {}
    with open(theories_path) as f:
        for line in f:
            if line.strip():
                theory = json.loads(line)["theory"]
                theories[theory_hash(theory)] = theory
    counts, examples = Counter(), {}
    with open(records_path) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            reason = check_record(record, theories.get(record["meta"].get("theory_hash")))
            counts[reason] += 1
            if reason != "agree" and len(examples.setdefault(reason, [])) < 5:
                examples[reason].append(record["id"])
    return counts, examples


def main(argv=None):
    ap = argparse.ArgumentParser(description="The Step 3 rule engine.")
    sub = ap.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="Derive the label and the depth of each record again and count the results.")
    check.add_argument("records")
    check.add_argument("--theories", help="theories.jsonl (default: theories.jsonl next to the records file)")
    args = ap.parse_args(argv)
    theories = args.theories or Path(args.records).with_name("theories.jsonl")
    counts, examples = check_files(args.records, theories)
    for reason, n in sorted(counts.items()):
        print(f"{reason}\t{n}\t{' '.join(examples.get(reason, []))}".rstrip())
    bad = sum(n for reason, n in counts.items() if reason != "agree")
    print(f"total\t{sum(counts.values())}\tnot agree: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
