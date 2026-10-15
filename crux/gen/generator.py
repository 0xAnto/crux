"""Step 3: the deterministic counterfactual generator (docs/generator_spec.md, sections 4 to 9 and 10.2).

The generator builds theories, computes its own labels with `solve` (it does not use crux.gen.engine for this),
renders the records, and asks the engine to check each record (section 9.3).
Run: python -m crux.gen.generator [--config configs/step3_generator.yaml] [--seed N] [--total N] [--out DIR]
"""
import argparse
import hashlib
import itertools
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from crux.gen.templates import FAMILIES, label as rule_label
from crux.schema import ROOT, write_jsonl

FEATURES = ("conj", "neg", "thr", "exc", "pri")
BITS = dict(zip(FEATURES, (1, 2, 4, 8, 16)))
LABELS = ("True", "False", "Unknown")
OPS = {">=": lambda v, k: v >= k, ">": lambda v, k: v > k, "<=": lambda v, k: v <= k,
       "<": lambda v, k: v < k, "==": lambda v, k: v == k}
COMPLEMENT = {">=": "<", ">": "<=", "<=": ">", "<": ">="}
EDIT_TYPES = ("fact_flip", "fact_remove", "fact_number", "threshold_shift", "exception_remove", "priority_swap",
              "rule_remove", "condition_negate")
EDIT_NEEDS = {"fact_number": "thr", "threshold_shift": "thr", "exception_remove": "exc", "priority_swap": "pri"}
MC_FEATURES = {"neg", "thr", "exc", "pri"}
HELD_OUT = {"val": ("c08", "c05", "c11", "c23"), "test": ("c18", "c28", "c31")}
COMPOSITIONS = {"train": tuple(c for c in ("c%02d" % b for b in range(32)) if c not in HELD_OUT["val"] + HELD_OUT["test"]),
                **HELD_OUT}
TEMPLATE_GROUPS = {"train": ("tg0", "tg1", "tg2"), "val": ("tg3",), "test": ("tg4",)}
VOCAB_IDS = {"train": tuple("v%02d" % i for i in range(8)), "val": ("v08", "v09"), "test": ("v10", "v11")}
VOCAB = json.loads(Path(__file__).with_name("vocab.json").read_text())
STOP = set("a an the is are not no has have does do of to for with in on at by from and or its it be been as per".split())
DENYLIST = ROOT / "tests" / "data" / "denylist_tokens.txt"
DEFAULT_CONFIG = ROOT / "configs" / "step3_generator.yaml"


def derive_seed(base_seed, *parts):
    key = "/".join(str(p) for p in (base_seed, *parts)).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def canonical(theory):
    return json.dumps(theory, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def theory_hash(theory):
    return hashlib.sha256(canonical(theory).encode()).hexdigest()[:16]


def comp_features(comp):
    bits = int(comp[1:])
    return [f for f in FEATURES if bits & BITS[f]]


# ---------------------------------------------------------------- own evaluator (sections 1 to 3)

def _fail(code, text):
    raise ValueError(f"{code}: {text}")


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _check_conds(conds, attrs, where):
    if not isinstance(conds, list) or not conds:
        _fail("schema", f"{where}: 'if' must be a non-empty list")
    seen = set()
    for c in conds:
        if not isinstance(c, dict) or set(c) not in ({"attr", "neg"}, {"attr", "op", "value"}):
            _fail("schema", f"{where}: bad condition {c!r}")
        a = c["attr"]
        if not isinstance(a, str):
            _fail("schema", f"{where}: bad attribute {a!r}")
        if a in seen:
            _fail("schema", f"{where}: two conditions on {a}")
        seen.add(a)
        if "neg" in c:
            if not isinstance(c["neg"], bool):
                _fail("schema", f"{where}: 'neg' must be a bool")
        elif not isinstance(c["op"], str) or c["op"] not in OPS or not _is_int(c["value"]) or not 0 <= c["value"] <= 99:
            _fail("schema", f"{where}: bad threshold {c!r}")
        if a not in attrs:
            _fail("ref", f"{where}: unknown attribute {a}")
        if ("neg" in c) != (attrs[a] == "bool"):
            _fail("type", f"{where}: condition form does not match the type of {a}")


def _check_theory(t):
    """Run the static checks of section 2.6. Return the derived attributes in a topological order."""
    keys = {"entities", "attributes", "facts", "rules", "exceptions", "priorities"}
    if not isinstance(t, dict) or set(t) != keys:
        _fail("schema", "a theory has exactly the keys " + ", ".join(sorted(keys)))
    ents, attrs = t["entities"], t["attributes"]
    if not isinstance(ents, list) or not ents or not all(isinstance(e, str) and re.fullmatch(r"e[0-9]+", e) for e in ents) \
            or len(set(ents)) != len(ents):
        _fail("schema", "bad entities")
    if not isinstance(attrs, dict) or not all(isinstance(a, str) and re.fullmatch(r"a[0-9]+", a) and v in ("bool", "int")
                                              for a, v in attrs.items()):
        _fail("schema", "bad attributes")
    for k in ("facts", "rules", "exceptions", "priorities"):
        if not isinstance(t[k], list) or not all(isinstance(x, dict) for x in t[k]):
            _fail("schema", f"'{k}' must be a list of objects")
    seen = set()
    for f in t["facts"]:
        if set(f) != {"entity", "attr", "value"} or not isinstance(f["entity"], str) or not isinstance(f["attr"], str):
            _fail("schema", f"bad fact {f!r}")
        if (f["entity"], f["attr"]) in seen:
            _fail("schema", f"duplicate fact {f['entity']} {f['attr']}")
        seen.add((f["entity"], f["attr"]))
        if f["entity"] not in ents or f["attr"] not in attrs:
            _fail("ref", f"fact {f!r}")
        if attrs[f["attr"]] == "bool" and not isinstance(f["value"], bool):
            _fail("type", f"fact {f!r} needs a bool")
        if attrs[f["attr"]] == "int":
            if not _is_int(f["value"]):
                _fail("type", f"fact {f!r} needs an int")
            if not 0 <= f["value"] <= 99:
                _fail("schema", f"fact {f!r} is out of 0 to 99")
    rules = {}
    for r in t["rules"]:
        if set(r) != {"id", "if", "then"} or not isinstance(r["id"], str) or not re.fullmatch(r"r[0-9]+", r["id"]) \
                or r["id"] in rules:
            _fail("schema", f"bad rule {r!r}")
        _check_conds(r["if"], attrs, r["id"])
        th = r["then"]
        if not isinstance(th, dict) or set(th) != {"attr", "neg"} or not isinstance(th["neg"], bool) \
                or not isinstance(th["attr"], str):
            _fail("schema", f"{r['id']}: bad 'then'")
        if th["attr"] not in attrs:
            _fail("ref", f"{r['id']}: unknown attribute {th['attr']}")
        if attrs[th["attr"]] != "bool":
            _fail("type", f"{r['id']}: 'then' on an int attribute")
        rules[r["id"]] = r
    xs = set()
    for x in t["exceptions"]:
        if set(x) != {"id", "if", "blocks"} or not isinstance(x["id"], str) or not re.fullmatch(r"x[0-9]+", x["id"]) \
                or x["id"] in xs or not isinstance(x["blocks"], str):
            _fail("schema", f"bad exception {x!r}")
        xs.add(x["id"])
        _check_conds(x["if"], attrs, x["id"])
        if x["blocks"] not in rules:
            _fail("ref", f"{x['id']} blocks an unknown rule")
    derived = {r["then"]["attr"] for r in rules.values()}
    for f in t["facts"]:
        if f["attr"] in derived:
            _fail("derived_fact", f"fact on the derived attribute {f['attr']}")
    pairs, after = set(), defaultdict(set)
    for p in t["priorities"]:
        if set(p) != {"higher", "lower"} or not isinstance(p["higher"], str) or not isinstance(p["lower"], str):
            _fail("schema", f"bad priority {p!r}")
        if p["higher"] not in rules or p["lower"] not in rules:
            _fail("ref", f"priority {p!r}")
        hi, lo = rules[p["higher"]]["then"], rules[p["lower"]]["then"]
        if p["higher"] == p["lower"] or hi["attr"] != lo["attr"] or hi["neg"] == lo["neg"] or (p["higher"], p["lower"]) in pairs:
            _fail("priority", f"bad pair {p!r}")
        pairs.add((p["higher"], p["lower"]))
        after[p["higher"]].add(p["lower"])
    if _topological(sorted(rules), after) is None:
        _fail("priority", "cycle in the priority graph")
    edges = defaultdict(set)
    for r in rules.values():
        for c in r["if"]:
            edges[c["attr"]].add(r["then"]["attr"])
    for x in t["exceptions"]:
        for c in x["if"]:
            edges[c["attr"]].add(rules[x["blocks"]]["then"]["attr"])
    order = _topological(sorted(attrs), edges)
    if order is None:
        _fail("cycle", "cycle in the attribute dependency graph")
    return [a for a in order if a in derived]


def _topological(nodes, edges):
    """Kahn's algorithm. Return the nodes in a topological order, or None for a cycle."""
    indeg = Counter()
    for n in nodes:
        for m in sorted(edges.get(n, ())):
            indeg[m] += 1
    ready = [n for n in nodes if not indeg[n]]
    order = []
    while ready:
        n = ready.pop()
        order.append(n)
        for m in sorted(edges.get(n, ())):
            indeg[m] -= 1
            if not indeg[m]:
                ready.append(m)
    return order if len(order) == len(nodes) else None


def _check_query(t, q):
    attrs = t["attributes"]
    if not isinstance(q, dict) or q.get("type") not in ("binary", "multi_choice"):
        _fail("query", "bad query type")
    if q["type"] == "binary":
        if set(q) != {"type", "entity", "attr", "neg"} or not isinstance(q["neg"], bool):
            _fail("query", "bad binary query keys")
        names = [q["attr"]]
    else:
        if set(q) != {"type", "entity", "candidates"} or not isinstance(q["candidates"], list):
            _fail("query", "bad multi-choice query keys")
        names = q["candidates"]
        if not 2 <= len(names) <= 8 or len(set(map(str, names))) != len(names):
            _fail("query", "a multi-choice query needs 2 to 8 different candidates")
    if not isinstance(q["entity"], str) or q["entity"] not in t["entities"]:
        _fail("query", f"unknown entity {q['entity']}")
    for a in names:
        if not isinstance(a, str) or attrs.get(a) != "bool":
            _fail("query", f"{a!r} is not a bool attribute")


def _model(t, order):
    """Compute val and d (section 3.1) for every entity and attribute. Raise 'conflict' for an unresolved conflict."""
    val, dep = {}, {}
    for f in t["facts"]:
        val[(f["entity"], f["attr"])] = f["value"]
        dep[(f["entity"], f["attr"])] = 0
    by_attr, by_rule = defaultdict(list), defaultdict(list)
    for r in t["rules"]:
        by_attr[r["then"]["attr"]].append(r)
    for x in t["exceptions"]:
        by_rule[x["blocks"]].append(x)
    beats = {(p["higher"], p["lower"]) for p in t["priorities"]}

    def holds(e, c):
        v = val.get((e, c["attr"]))
        if v is None:
            return False
        return v == (not c["neg"]) if "neg" in c else OPS[c["op"]](v, c["value"])

    for a in order:
        for e in t["entities"]:
            fire = [r for r in by_attr[a] if all(holds(e, c) for c in r["if"])
                    and not any(all(holds(e, c) for c in x["if"]) for x in by_rule[r["id"]])]
            pos = [r for r in fire if not r["then"]["neg"]]
            neg = [r for r in fire if r["then"]["neg"]]
            if not fire:
                continue
            if pos and neg:
                win_pos = [r for r in pos if all((r["id"], s["id"]) in beats for s in neg)]
                win_neg = [r for r in neg if all((r["id"], s["id"]) in beats for s in pos)]
                if not win_pos and not win_neg:
                    _fail("conflict", f"unresolved conflict on {e} {a}")
                win, v = (win_pos, True) if win_pos else (win_neg, False)
            else:
                win, v = fire, bool(pos)
            val[(e, a)] = v
            dep[(e, a)] = min(1 + max(0 if "op" in c else dep[(e, c["attr"])] for c in r["if"]) for r in win)
    return val, dep


def _heights(t, order):
    h = {a: 0 for a in t["attributes"]}
    by_attr = defaultdict(list)
    for r in t["rules"]:
        by_attr[r["then"]["attr"]].append(r)
    for a in order:
        h[a] = max(1 + max(h[c["attr"]] for c in r["if"]) for r in by_attr[a])
    return h


def solve(theory, query):
    """The generator's own evaluator: {"label": str, "depth": int}. Raise ValueError for an invalid theory or query."""
    return _solve(theory, query)[0]


def _solve(theory, query):
    """Return (solve result, val)."""
    order = _check_theory(theory)
    _check_query(theory, query)
    val, dep = _model(theory, order)
    e = query["entity"]
    if query["type"] == "multi_choice":
        true = [c for c in query["candidates"] if val.get((e, c)) is True]
        if len(true) != 1:
            _fail("mc_answer", f"{len(true)} true candidates")
        return {"label": true[0], "depth": dep[(e, true[0])]}, val
    v = val.get((e, query["attr"]))
    if v is None:
        return {"label": "Unknown", "depth": _heights(theory, order)[query["attr"]]}, val
    return {"label": "True" if v != query["neg"] else "False", "depth": dep[(e, query["attr"])]}, val


def _cone(t, query):
    """The derived attributes that the query attributes depend on (through rules and exceptions), and the query
    attributes."""
    rules = {r["id"]: r for r in t["rules"]}
    up = defaultdict(set)
    for r in t["rules"]:
        up[r["then"]["attr"]].update(c["attr"] for c in r["if"])
    for x in t["exceptions"]:
        up[rules[x["blocks"]]["then"]["attr"]].update(c["attr"] for c in x["if"])
    todo = [query["attr"]] if query["type"] == "binary" else list(query["candidates"])
    cone = set(todo)
    while todo:
        for b in up[todo.pop()] - cone:
            cone.add(b)
            todo.append(b)
    derived = {r["then"]["attr"] for r in t["rules"]}
    return sorted(cone & derived)


def features(theory):
    """The generator's own copy of section 4.1."""
    conds = [c for r in theory["rules"] + theory["exceptions"] for c in r["if"]]
    present = {"conj": any(len(r["if"]) >= 2 for r in theory["rules"] + theory["exceptions"]),
               "neg": any(c.get("neg") is True for c in conds),
               "thr": any("op" in c for c in conds),
               "exc": bool(theory["exceptions"]), "pri": bool(theory["priorities"])}
    return [f for f in FEATURES if present[f]]


# ---------------------------------------------------------------- plan (section 9.2)

def make_plan(cfg, seed, k, flip_p=None, label_w=None, pos_w=None, edit_w=None, type_w=None):
    """Draw the plan of unit k with plan_rng, in the order of section 9.2. flip_p: the plan probability of a flip
    (default flip_share). label_w: the weights of the binary labels (default uniform). pos_w: option count -> the
    weights of the answer positions (default uniform). edit_w: label -> the weight of the edit label (default
    uniform). type_w: edit type -> the weight of the planned edit type (default uniform)."""
    r = random.Random(derive_seed(seed, "plan", k))
    u, sh = r.random(), cfg["block_shares"]
    block = "train" if u < sh["train"] else "val" if u < sh["train"] + sh["val"] else "test"
    tg = r.choice(TEMPLATE_GROUPS[block])
    n_t = r.choices((1, 2, 3), weights=cfg["templates_per_unit"])[0]
    templates = r.sample(["t%02d" % (3 * int(tg[2:]) + i) for i in range(3)], n_t)
    vocab = r.choice(VOCAB_IDS[block])
    comp = r.choice(COMPOSITIONS[block])
    feats = comp_features(comp)
    # spec: section 9.2. A multi-choice flip needs one edit that moves the answer to another option, so it needs
    # neg, thr, exc, or pri, and depth 1 or more (one fact edit cannot turn one base option off and another on).
    # Compositions without these features have no multi-choice units; the others have more, so that the block share
    # stays at multi_choice_share.
    ok = [c for c in COMPOSITIONS[block] if MC_FEATURES & set(comp_features(c))]
    m = cfg["multi_choice_share"] * len(COMPOSITIONS[block]) / len(ok) if comp in ok else 0.0
    mc = comp in ok and r.random() < m
    w = [x / sum(cfg["depth_weights"][block]) for x in cfg["depth_weights"][block]]
    # spec: section 9.2. A multi-choice unit has depth 1 or more, so a binary unit takes the depth-0 share of the
    # multi-choice units too: the depth of all units of the composition follows depth_weights.
    wb = [w[0] / (1 - m)] + [x * (1 - m / (1 - w[0])) / (1 - m) for x in w[1:]]
    depth = r.choices(range(1, len(w)), weights=w[1:])[0] if mc else r.choices(range(len(w)), weights=wb)[0]
    # spec: section 5.2. The word cap limits the number of options of a deep unit.
    n_max = cfg["max_options"]["conj" if "conj" in feats else "single"][depth]
    n_opt = r.randint(cfg["option_counts"][0], n_max) if mc else 0
    flip = r.random() < (cfg["flip_share"] if flip_p is None else flip_p)
    # spec: section 9.2. A binary depth-0 Unknown cannot flip (no edit adds a fact), so a depth-0 flip has the label
    # True or False.
    lw = list(label_w or (1, 1, 1))
    if depth == 0 and flip:
        lw[2] = 0
    lab = r.choices(range(n_opt), weights=(pos_w or {}).get(n_opt, [1] * n_opt))[0] if mc else \
        r.choices(LABELS, weights=lw)[0]
    pe = permitted_edits(feats)
    edit_type = r.choices(pe, weights=[(type_w or {}).get(t, 1) for t in pe])[0]
    neg = (not mc) and r.random() < cfg["binary_neg_share"]
    # spec: section 9.2. The edit label: a binary unit prefers an edit group with a flip to this label, for a flip
    # and for a stay, so that the edit labels stay balanced and the group does not show the flip status.
    # At depth 0, a fact_flip of the query fact turns True into False. At depth 1 or more, a True <-> False flip
    # needs pri, or a twin (neg or thr, with conj or exc: section 9.5). Else a True or False unit has the edit label
    # Unknown.
    # An Unknown unit can flip to True or False (both are known), so it has no edit label.
    fs = set(feats)
    known = depth == 0 or "pri" in fs or bool({"neg", "thr"} & fs and {"conj", "exc"} & fs)
    allowed = [x for x in LABELS if x != lab and (x == "Unknown" or known)]
    edit_label = None if mc or lab == "Unknown" else \
        r.choices(allowed, weights=[(edit_w or {}).get(x, 1) for x in allowed])[0]
    return {"block": block, "template_group": tg, "templates": templates, "vocab": vocab, "composition": comp,
            "features": feats, "task": "multi_choice" if mc else "binary", "n_options": n_opt, "depth": depth,
            "label": lab, "flip": flip, "edit_type": edit_type, "neg": neg, "edit_label": edit_label}


def permitted_edits(feats):
    return [t for t in EDIT_TYPES if t not in EDIT_NEEDS or EDIT_NEEDS[t] in feats]


# ---------------------------------------------------------------- theory construction (section 9.5)

class _Clash(Exception):
    """Two requirements of the fact search disagree."""


class _Builder:
    """Build one theory and query for a plan (one attempt), in two stages (section 9.5).
    1. The structure (attributes, rules, exceptions, priorities) uses only the block, the features, the task, the
       number of options, the depth, and the query polarity. It does not use the planned label, flip, or edit type.
    2. The fact search chooses the facts of the query entity so that the planned label (binary) or the planned
       option (multi-choice) holds. Any option of a structure can be the answer, so the structure does not show it."""

    def __init__(self, plan, rng, cfg, lean=0):
        self.p, self.rng, self.F, self.cfg, self.lean = plan, rng, set(plan["features"]), cfg, lean
        self.maxh = cfg["max_height"][plan["block"]]
        self.ents = ["e%d" % i for i in range(rng.randint(*cfg["entities"]))]
        self.q = rng.choice(self.ents)
        self.attrs, self.h, self.base = {}, {}, []
        self.rules, self.exceptions, self.priorities = [], [], []
        self.facts = {}
        self.ids = itertools.count()

    # -- small parts
    def attr(self, typ, base=True):
        a = "a%d" % next(self.ids)
        self.attrs[a], self.h[a] = typ, 0
        if base and typ == "bool":
            self.base.append(a)
        return a

    def rule(self, conds, attr, neg):
        r = {"id": "r%d" % next(self.ids), "if": conds, "then": {"attr": attr, "neg": neg}}
        self.rules.append(r)
        self.h[attr] = max(self.h[attr], 1 + max(self.h[c["attr"]] for c in conds))
        return r

    def exception(self, conds, rule):
        self.exceptions.append({"id": "x%d" % next(self.ids), "if": conds, "blocks": rule["id"]})

    def priority(self, hi, lo):
        self.priorities.append({"higher": hi["id"], "lower": lo["id"]})

    def base_lit(self, avoid=(), kind=None):
        """A condition on a base attribute (not in `avoid`). Base attributes are often used again, so that one fact
        can matter for more than one rule."""
        rng, F = self.rng, self.F
        if kind is None:
            kind = "thr" if "thr" in F and rng.random() < 0.35 else "bool"
        if kind == "thr":
            ints = [a for a, t in self.attrs.items() if t == "int" and a not in avoid]
            n_int = sum(t == "int" for t in self.attrs.values())
            x = rng.choice(ints) if ints and (n_int >= 4 or rng.random() < 0.5) else None
            if x is None:
                if n_int >= 4:
                    return self.base_lit(avoid, "bool")
                x = self.attr("int")
            return {"attr": x, "op": rng.choice(("<", "<=", ">", ">=")), "value": rng.randint(10, 89)}
        pool = [a for a in self.base if a not in avoid]
        a = rng.choice(pool) if pool and rng.random() < self.cfg["share_base"] else self.attr("bool")
        return {"attr": a, "neg": "neg" in F and rng.random() < 0.35}

    def extras(self, conds, avoid=()):
        """0 to 2 more base conditions (only with conj), not on the attributes in `avoid`."""
        if "conj" not in self.F:
            return conds
        n = self.rng.choice((0, 0, 1, 1, 2) if not self.lean else (0, 0, 1))
        for _ in range(min(n, self.cfg["max_conditions"] - len(conds))):
            conds.append(self.base_lit({c["attr"] for c in conds} | set(avoid)))
        return conds

    def level_lit(self, a, neg):
        return {"attr": a, "neg": neg}

    # -- the structure
    def chain(self, D, bottom=None):
        """Levels 1 .. D-1 of a chain. Return the literal that a rule of level D reads (a base literal for D = 1).
        The main rule of a level concludes the value that the next level reads. An opposite rule (with a base
        condition) can only break the chain; it never proves a value at a lower depth (section 3.1)."""
        rng, F = self.rng, self.F
        lit = bottom if bottom is not None else self.base_lit()
        for k in range(1, D):
            a = self.attr("bool", base=False)
            read_neg = "neg" in F and rng.random() < 0.3
            main = self.rule(self.extras([lit]), a, read_neg)
            if not self.lean and rng.random() < (0.6 if "pri" in F else 0.0):
                # From level 2, the opposite rule can read a derived side value (the height stays k): the removal of
                # the side rule then repairs the chain (section 6, pair coverage of rule_remove).
                first = self.side() if k >= 2 and rng.random() < 0.4 else self.base_lit()
                opp = self.rule(self.extras([first]), a, not read_neg)
                self.priority(main, opp) if rng.random() < 0.5 else self.priority(opp, main)
            lit = self.level_lit(a, read_neg)
        return lit

    def side(self):
        """A derived side attribute of height 1, read as an extra condition (a change of it is a relevant stay)."""
        s = self.attr("bool", base=False)
        neg = "neg" in self.F and self.rng.random() < 0.3
        self.rule(self.extras([self.base_lit()]), s, neg)
        return self.level_lit(s, neg)

    def add_side(self, conds):
        """A side condition (height 1) only on a rule that already reads a derived value, so that the height and the
        depth of the rule stay the same."""
        if "conj" in self.F and not self.lean and len(conds) < self.cfg["max_conditions"] and \
                max(self.h[c["attr"]] for c in conds) >= 1 and self.rng.random() < 0.3:
            conds.append(self.side())
        return conds

    def binary_structure(self, D):
        rng, F = self.rng, self.F
        q_attr = self.attr("bool", base=False)
        if D == 0:
            self.base.append(q_attr)
            self.side_rules()
            return q_attr
        # spec: section 9.2. With neg or thr, P and N often read complementary literals of one base attribute (as own
        # conditions with conj, or in their exceptions with exc), so that one fact edit can turn True into False and
        # back. Without this, most binary flips end in Unknown.
        # spec: only with conj or exc. With one condition per rule, a twin leaves Unknown only for a missing fact,
        # and no edit adds a fact, so an Unknown unit with a twin has no flip there. A twin in some theories only
        # would then show the label (the accepted Unknown units would have none).
        alone = "conj" not in F and "exc" not in F
        tw = self.twin() if ("neg" in F or "thr" in F) and not alone else None
        top = self.chain(D)
        # N must reach depth D too (else the label False shows a lower depth). Something must separate N from P:
        # own base conditions (conj), exceptions (exc), or else its own chain.
        if "conj" in F:
            if tw and tw[0]["attr"] == top["attr"]:
                tw = None
            p = self.rule(self.add_side(self.own([top], tw and tw[0])), q_attr, False)
            n = self.rule(self.add_side(self.own([dict(top)], tw and tw[1])), q_attr, True)
        elif "exc" in F:
            p = self.rule([top], q_attr, False)
            n = self.rule([dict(top)], q_attr, True)
            for r, lit in zip((p, n), tw or (self.base_lit(), self.base_lit())):
                self.exception([lit], r)
        else:
            p = self.rule([top], q_attr, False)
            n = self.rule([self.chain(D)], q_attr, True)
        if "pri" in F:
            self.priority(p, n) if rng.random() < 0.5 else self.priority(n, p)
        if "exc" in F:
            for _ in range(1 if self.lean or rng.random() < 0.7 else 2):
                self.exception(self.extras([self.base_lit()])[:2], rng.choice(self.rules))
        self.side_rules(1)
        return q_attr

    def own(self, conds, lit=None):
        """At least one own base condition after the shared level (conj), so that P and N can differ."""
        conds.append(lit or self.base_lit({c["attr"] for c in conds}))
        return self.extras(conds)

    def twin(self):
        """A base literal and its complement on the same attribute (s and not s, x > k and x <= k), or None."""
        kinds = [k for k, f in (("bool", "neg"), ("thr", "thr")) if f in self.F]
        b = self.base_lit(kind=self.rng.choice(kinds))
        if "op" in b:
            return b, {**b, "op": COMPLEMENT[b["op"]]}
        return (b, {**b, "neg": not b["neg"]}) if "neg" in self.F else None

    def mc_structure(self, D, n):
        """The options in pairs (section 9.2): the two options of a pair react to one shared base fact in opposite
        ways (s and not s; x <= k and x > k; an exception or a stronger opposite rule on the partner that reads s).
        Thus one edit of that fact can move the answer to the partner, whichever option is the answer."""
        rng, F = self.rng, self.F
        cands = [self.attr("bool", base=False) for _ in range(n)]
        order = rng.sample(range(n), n)
        partner = {}
        for i in range(0, n - 1, 2):
            partner[order[i]], partner[order[i + 1]] = order[i + 1], order[i]
        ways = {i: rng.choice(sorted(MC_FEATURES & F)) for i in partner if i < partner[i]}
        top = self.chain(D) if "conj" in F else None
        trunk = {c["attr"] for r in self.rules for c in r["if"]} | ({top["attr"]} if top else set())  # an option
        # condition on them would hold with the answer (at depth 1 the trunk is the base literal `top`)
        owns, own = [], {}
        for i in range(n):
            j = partner.get(i)
            way = ways.get(min(i, j)) if j is not None else None
            if j is not None and j in own and way in ("neg", "thr"):
                o = own[j]
                own[i] = ({"attr": o["attr"], "op": COMPLEMENT[o["op"]], "value": o["value"]} if "op" in o else
                          {"attr": o["attr"], "neg": not o["neg"]})
            else:
                kind = "thr" if way == "thr" else "bool" if way == "neg" else None
                own[i] = self.base_lit(trunk | {o["attr"] for o in owns}, kind)
            owns.append(own[i])
        contest = {}
        for i, j in partner.items():
            if i < j and ways[i] == "pri" and D >= 2 and not contest and list(self.attrs.values()).count("bool") <= 11:
                # spec: section 9.5. A pair on a derived attribute d: option i reads d, and a rule "d -> not j" beats
                # the own rule of j. d has one rule, or a positive and a negative rule with a priority. Thus a rule
                # edit on the rules of d (rule_remove, condition_negate, threshold_shift, priority_swap) can move the
                # answer. d has height 1, so the chain of i is one level shorter (thus not at depth 1). One pair for
                # each unit, and only with at most 11 bool attributes so far: the pair adds up to 3, and the rest of
                # a large unit then often goes beyond the pool size (section 4.3).
                d = self.attr("bool", base=False)
                avoid = trunk | {o["attr"] for o in owns}
                pos = self.rule([self.base_lit(avoid)], d, False)
                if rng.random() < 0.5:
                    neg = self.rule([self.base_lit(avoid)], d, True)
                    self.priority(pos, neg) if rng.random() < 0.5 else self.priority(neg, pos)
                own[i], contest[i] = self.level_lit(d, False), j
        tops = {}
        for i, c in enumerate(cands):
            if top is not None:
                conds = [top, own[i]]
            elif i in contest:
                conds = [self.chain(D - 1, bottom=own[i])]
            else:
                conds = [self.chain(D, bottom=own[i])]
            tops[i] = self.rule(self.add_side(self.extras(conds, trunk)), c, False)
        for i, j in contest.items():
            self.priority(self.rule([dict(own[i])], cands[j], True), tops[j])
        for i, j in partner.items():
            way = ways[min(i, j)]
            if i < j and way in ("exc", "pri") and i not in contest:
                s = own[i] if "neg" in own[i] else own[j] if "neg" in own[j] else None
                if s is None:
                    continue
                k = j if s is own[i] else i  # the partner of the option that owns s
                if way == "exc":
                    self.exception([dict(s)], tops[k])
                else:
                    opp = self.rule([dict(s)], cands[k], True)
                    self.priority(opp, tops[k])
        for i, c in enumerate(cands):  # more exceptions and priorities, for variety
            if "exc" in F and rng.random() < 0.15:
                self.exception([self.base_lit()], tops[i])
            if "pri" in F and rng.random() < 0.15:
                opp = self.rule([self.base_lit()], c, True)
                self.priority(tops[i], opp) if rng.random() < 0.5 else self.priority(opp, tops[i])
        self.side_rules(1)
        return cands

    def fit(self, c):
        """Make a condition fit the type of its attribute (`pick` can move a condition to another attribute)."""
        if self.attrs[c["attr"]] == "int":
            return {"attr": c["attr"], "op": c.get("op") or self.rng.choice(("<", "<=", ">", ">=")),
                    "value": c.get("value", self.rng.randint(10, 89))}
        return {"attr": c["attr"], "neg": "neg" in self.F and self.rng.random() < 0.35}

    def side_rules(self, lo=1):
        """Distractor rules: they do not lead to the query attributes. They use the same base attributes, so an edit
        can change a conclusion about the entity without a change of the answer (section 6, relevant stays)."""
        rng = self.rng
        levels = [r["then"]["attr"] for r in self.rules if self.h[r["then"]["attr"]] < self.maxh]
        for k in range(rng.randint(lo, (1 if self.lean else 3) if lo else (0 if self.lean else 2))):
            s = self.attr("bool", base=False)
            used = [c for r in self.rules + self.exceptions for c in r["if"] if self.h[c["attr"]] == 0]
            u = rng.random()
            if k == 0 and list(self.attrs.values()).count("bool") <= 20:
                # A new base attribute, so that q has a free fact. Not near the pool size (section 4.3).
                first = {"attr": self.attr("bool"), "neg": "neg" in self.F and rng.random() < 0.35}
            elif levels and u < 0.25:
                first = self.level_lit(rng.choice(levels), "neg" in self.F and rng.random() < 0.3)
            elif used and u < 0.5:  # a fact that the query already uses
                first = self.fit({"attr": rng.choice(used)["attr"]})
            else:  # a fresh fact: its edit changes the side value and not the answer (a relevant stay of a fact edit)
                first = self.base_lit(kind="thr" if "thr" in self.F and rng.random() < 0.5 else "bool")
            side = self.rule(self.extras([first]), s, rng.random() < 0.3)
            # An exception or a stronger opposite rule on a side rule: its removal or swap can change the side
            # value without a change of the answer (a relevant stay of exception_remove and priority_swap).
            if "exc" in self.F and not self.lean and rng.random() < 0.5:
                self.exception([self.fit({"attr": rng.choice(used)["attr"]}) if used and rng.random() < 0.7
                                else self.base_lit()], side)
            if "pri" in self.F and not self.lean and rng.random() < 0.5:
                opp = self.rule([self.fit({"attr": rng.choice(used)["attr"]}) if used and rng.random() < 0.7
                                 else self.base_lit()], s, not side["then"]["neg"])
                self.priority(side, opp) if rng.random() < 0.5 else self.priority(opp, side)

    def complete_features(self):
        """Add the parts that a feature of the composition needs, if the random structure does not have them."""
        rng, F = self.rng, self.F
        rules = self.rules
        if not rules:
            s = self.attr("bool", base=False)
            self.rule([self.base_lit()], s, False)
            rules = self.rules
        if "conj" in F and not any(len(r["if"]) >= 2 for r in rules + self.exceptions):
            r = rng.choice(rules)
            r["if"].append(self.base_lit({c["attr"] for c in r["if"]}))
        if "neg" in F and not any(c.get("neg") for r in rules + self.exceptions for c in r["if"]):
            bools = [c for r in rules + self.exceptions for c in r["if"] if "neg" in c and self.h[c["attr"]] == 0]
            if not bools:
                return False
            rng.choice(bools)["neg"] = True
        if "thr" in F and not any("op" in c for r in rules + self.exceptions for c in r["if"]):
            spots = [(r, i) for r in rules + self.exceptions for i, c in enumerate(r["if"])
                     if "neg" in c and self.h[c["attr"]] == 0]
            if not spots:
                return False
            r, i = rng.choice(spots)
            r["if"][i] = self.base_lit({c["attr"] for c in r["if"]}, "thr")
        if "exc" in F and not self.exceptions:
            self.exception(self.extras([self.base_lit()])[:2], rng.choice(self.rules))
        if "pri" in F and not self.priorities:
            r = rng.choice(self.rules)
            opp = self.rule(self.extras([self.base_lit()]), r["then"]["attr"], not r["then"]["neg"])
            self.priority(r, opp) if rng.random() < 0.5 else self.priority(opp, r)
        for r in self.rules:  # recompute the heights (complete_features can change conditions)
            self.h[r["then"]["attr"]] = 0
        for _ in range(len(self.rules)):
            for r in self.rules:
                self.h[r["then"]["attr"]] = max(self.h[r["then"]["attr"]], 1 + max(self.h[c["attr"]] for c in r["if"]))
        return True

    # -- the fact search (stage 2)
    def search(self, target):
        """Choose the facts of q. target: ("value", attr, v) for each attribute in a list. Return True on success."""
        rng = self.rng
        self.by_attr, self.by_rule = defaultdict(list), defaultdict(list)
        for r in self.rules:
            self.by_attr[r["then"]["attr"]].append(r)
        for x in self.exceptions:
            self.by_rule[x["blocks"]].append(x)
        self.beats = {(x["higher"], x["lower"]) for x in self.priorities}
        for _ in range(12):
            self.fix, self.made = {}, {}
            try:
                for a, v in target:
                    if v != "not_true":
                        self.make(a, v)
                        continue
                    for w in rng.sample([None, False], 2):  # an option that is not the answer: unknown or false
                        snap = self.snapshot()
                        try:
                            self.make(a, w)
                            break
                        except _Clash:
                            self.restore(snap)
                    else:
                        raise _Clash
            except _Clash:
                continue
            return True
        return False

    def snapshot(self):
        return dict(self.fix), dict(self.made)

    def restore(self, snap):
        self.fix, self.made = dict(snap[0]), dict(snap[1])

    def setv(self, a, v):
        if a in self.fix and self.fix[a] != v:
            raise _Clash
        self.fix[a] = v

    def hold(self, c):
        a = c["attr"]
        if self.by_attr.get(a):
            return self.make(a, not c["neg"])
        if "op" in c:
            v = self.fix.get(a, "free")
            if v == "free":
                ok = [w for w in range(100) if OPS[c["op"]](w, c["value"])]
                return self.setv(a, self.rng.choice(ok))
            if v is None or not OPS[c["op"]](v, c["value"]):
                raise _Clash
            return
        self.setv(a, not c["neg"])

    def fail(self, c):
        rng, a = self.rng, c["attr"]
        if self.by_attr.get(a):
            # spec: an opposite rule that wins breaks the chain more often than a missing value (0.7), so that a
            # priority swap, a rule removal, or a fact removal can repair the chain (section 6, pair coverage).
            # If the first value clashes, try the other one (a contested pair needs an unknown value).
            opp = any(r["then"]["neg"] == (not c["neg"]) for r in self.by_attr[a])
            for v in ([c["neg"], None] if opp and rng.random() < 0.7 else [None, c["neg"]] if opp else [None]):
                snap = self.snapshot()
                try:
                    return self.make(a, v)
                except _Clash:
                    self.restore(snap)
            raise _Clash
        missing = rng.random() < 0.35
        v = self.fix.get(a, "free")
        if "op" in c:
            if v == "free":
                bad = [w for w in range(100) if not OPS[c["op"]](w, c["value"])]
                return self.setv(a, None if missing or not bad else rng.choice(bad))
            if v is not None and OPS[c["op"]](v, c["value"]):
                raise _Clash
            return
        if v == "free":
            return self.setv(a, None if missing else c["neg"])
        if v == (not c["neg"]):  # the fixed value makes the literal hold
            raise _Clash

    def stop(self, r):
        """Rule r does not fire for q: one condition fails (the others hold, if they can), or an exception on r is
        active."""
        rng = self.rng
        xs = self.by_rule.get(r["id"], ())
        x = rng.choice(xs) if xs else None
        # spec: section 9.5. An active exception with probability 1/(m+1) (m: its conditions), and the conditions of r
        # hold too: k + m holding conditions (k: the conditions of r). Else one condition of r fails: k - 1 holding
        # conditions, and the m conditions of the exception are not set. Counted over the conditions of r and of the
        # exception, the mean is k, as for a firing rule. The walk (`walk`) draws these facts again afterwards.
        if x and rng.random() < 1 / (len(x["if"]) + 1):
            snap = self.snapshot()
            try:
                for c in r["if"] + x["if"]:
                    self.hold(c)
                return
            except _Clash:
                self.restore(snap)
        order = list(r["if"])
        rng.shuffle(order)
        order.sort(key=lambda c: not self.fails_now(c))  # a condition that already fails needs no new fact
        for i, c in enumerate(order):
            snap = self.snapshot()
            try:
                self.fail(c)
                for d in order:  # the other conditions hold, so that a stopped rule differs from a firing one by one fact
                    if d is not c:
                        snap2 = self.snapshot()
                        try:
                            self.hold(d)
                        except _Clash:
                            self.restore(snap2)
                return
            except _Clash:
                self.restore(snap)
        raise _Clash

    def fails_now(self, c):
        a = c["attr"]
        if self.by_attr.get(a):
            return a in self.made and self.made[a] != (not c["neg"])
        v = self.fix.get(a, "free")
        if v == "free":
            return False
        if v is None:
            return True
        return not OPS[c["op"]](v, c["value"]) if "op" in c else v != (not c["neg"])

    def make(self, a, v):
        """Attribute a gets the value v (True, False, or None for unknown) for q."""
        if a in self.made:
            if self.made[a] != v:
                raise _Clash
            return
        self.made[a] = v
        rules = list(self.by_attr.get(a, ()))
        self.rng.shuffle(rules)
        if v is None:
            for r in rules:
                self.stop(r)
            return
        win = [r for r in rules if r["then"]["neg"] == (not v)]
        if not win:
            raise _Clash
        r = win[0]
        for c in r["if"]:
            self.hold(c)
        for x in self.by_rule.get(r["id"], ()):
            self.fail(self.rng.choice(x["if"]))
        for s in rules:
            if s is not r and s["then"]["neg"] != r["then"]["neg"] and (r["id"], s["id"]) not in self.beats:
                self.stop(s)

    def fill(self):
        """The facts: the fixed values of q, random values for the other base attributes of q, and noise facts of the
        other entities."""
        rng, q = self.rng, self.q
        heads = {r["then"]["attr"] for r in self.rules}
        used = sorted(({c["attr"] for r in self.rules + self.exceptions for c in r["if"]} | set(self.fix)) - heads,
                      key=lambda x: int(x[1:]))
        self.facts = {(q, b): v for b, v in self.fix.items() if v is not None and b not in heads}
        # spec: section 9.5. The free base attributes of q take random values with the fixed FACT_SHARES, independent
        # of the label. An earlier version moved the totals of q toward fixed shares with the free facts: the free
        # facts then leaned against the fixed ones, so the facts off the query path showed the label.
        for b in sorted(self.attrs, key=lambda x: int(x[1:])):
            if b not in heads and self.fix.get(b, "free") == "free":
                shares = FACT_SHARES[self.attrs[b]]
                v = rng.choices(list(shares), weights=list(shares.values()))[0]
                if v is not None:
                    self.facts[(q, b)] = rng.randint(0, 99) if v == "v" else v
        share = 0.5 / (1 + self.lean)
        for e in self.ents:
            if e != q:
                for b in used:
                    if rng.random() < share:
                        self.facts[(e, b)] = rng.random() < 0.5 if self.attrs[b] == "bool" else rng.randint(0, 99)

    def words(self):
        """An estimate of the longest instruction (words) over the templates of the unit (section 5.2). Each
        attribute phrase counts with the mean phrase length of the pool."""
        pool = VOCAB[self.p["vocab"]]
        mp, mn = (sum(len(x[i].split()) for x in pool["bool"]) / len(pool["bool"]) for i in (0, 1))
        lex = dict(zip(self.ents, pool["names"]))
        lex.update({a: ("Ⓟ", "Ⓝ") if typ == "bool" else pool["int"][0] for a, typ in self.attrs.items()})
        b = next(a for a, typ in self.attrs.items() if typ == "bool")
        query = {"type": "binary", "entity": self.q, "attr": b, "neg": True}
        out = 0
        for tid in self.p["templates"]:
            text = render(self.theory(), query, lex, self.p["vocab"], tid)[1]
            out = max(out, len(text.split()) + text.count("Ⓟ") * (mp - 1) + text.count("Ⓝ") * (mn - 1))
        return out

    def theory(self):
        return {"entities": self.ents, "attributes": self.attrs,
                "facts": [{"entity": e, "attr": a, "value": v} for (e, a), v in self.facts.items()],
                "rules": self.rules, "exceptions": self.exceptions, "priorities": self.priorities}

    # -- the full build
    def build(self):
        """Return (theory, query) that match the plan, or None, or "long" (the word budget is exceeded)."""
        p, rng, q, cfg = self.p, self.rng, self.q, self.cfg
        D, mc, n = p["depth"], p["task"] == "multi_choice", p["n_options"]
        if mc:
            cands = self.mc_structure(D, n)
        else:
            a = self.binary_structure(D)
        if not self.complete_features():
            return None
        types = Counter(self.attrs.values())
        if types["bool"] > 24 or types["int"] > 4:  # the pool sizes (section 4.3)
            return None
        if self.lean < 3 and self.words() > cfg["caps"]["instruction_words"] - cfg["word_margin"]:
            return "long"
        if mc:
            ans = cands[p["label"]]
            target = [(ans, True)] + [(c, "not_true") for c in cands if c != ans]
        else:
            v = {"True": not p["neg"], "False": p["neg"], "Unknown": None}[p["label"]]
            target = [(a, v)]
        if not mc and D >= 1:
            # spec: section 9.5. Accept a binary structure only if the fact search succeeds for all three labels. Else
            # a structure that only one label permits (e.g. P and N read the same literal: only Unknown) is accepted
            # for that label only, and the structure shows the label.
            for lab, w in (("True", not p["neg"]), ("False", p["neg"]), ("Unknown", None)):
                if lab != p["label"] and not self.search([(a, w)]):
                    return None
        for _ in range(4):
            if not self.search_base(target, D == 0):
                return None
            for _ in range(3):
                self.fill()
                for _ in self.ents:  # a noise fact of another entity can cause a conflict: drop its facts
                    theory = self.theory()
                    try:
                        _model(theory, _check_theory(theory))
                        break
                    except ValueError as e:
                        m = re.search(r"on (e[0-9]+) ", str(e))
                        if not m or m.group(1) == q:
                            break
                        self.facts = {k: v for k, v in self.facts.items() if k[0] != m.group(1)}
                query = ({"type": "multi_choice", "entity": q, "candidates": list(cands)} if mc else
                         {"type": "binary", "entity": q, "attr": a, "neg": p["neg"]})
                try:
                    res = solve(theory, query)
                except ValueError:
                    continue
                want = cands[p["label"]] if mc else p["label"]
                if res["label"] == want and res["depth"] == D and theory["facts"]:  # the state is never empty
                    self.walk(query, want, D)
                    return self.finish(query)
        return None

    def walk(self, query, want, D, sweeps=4):
        """Metropolis-Hastings on the base facts of q that a condition reads. A move draws a new value with
        FACT_SHARES and stays only if the answer and the depth stay. The proposal is the prior, so the facts tend to
        the label-blind distribution for the answer and the depth. This holds for the built theories: `choose_edit`
        then keeps only theories with a balanced edit, and that shifts the facts of the accepted units (an Unknown
        theory with many missing facts often has no flip, because no edit adds a fact)."""
        # spec: section 9.5. The search alone made the facts on the query path depend on the label in a different
        # way in each composition (an active exception needs true facts), so a fact-count pattern of the train
        # compositions was inverted in c08. ponytail: a fixed number of sweeps, not a convergence test.
        rng, q = self.rng, self.q
        heads = {r["then"]["attr"] for r in self.rules}
        read = sorted({c["attr"] for x in self.rules + self.exceptions for c in x["if"]} - heads, key=lambda x: int(x[1:]))
        for _ in range(sweeps):
            for b in rng.sample(read, len(read)):
                old = self.facts.get((q, b))
                shares = FACT_SHARES[self.attrs[b]]
                v = rng.choices(list(shares), weights=list(shares.values()))[0]
                v = rng.randint(0, 99) if v == "v" else v
                if v == old:
                    continue  # no change: the state is valid
                self.facts.pop((q, b), None)
                if v is not None:
                    self.facts[(q, b)] = v
                try:
                    res = solve(self.theory(), query)
                    ok = res["label"] == want and res["depth"] == D and self.facts
                except ValueError:
                    ok = False
                if not ok:
                    self.facts.pop((q, b), None)
                    if old is not None:
                        self.facts[(q, b)] = old

    def search_base(self, target, base_only):
        """Depth 0: the query attributes are base attributes, so the fact search sets them directly."""
        if not base_only:
            return self.search(target)
        self.by_attr, self.by_rule = defaultdict(list), defaultdict(list)
        for r in self.rules:
            self.by_attr[r["then"]["attr"]].append(r)
        for x in self.exceptions:
            self.by_rule[x["blocks"]].append(x)
        self.beats = {(x["higher"], x["lower"]) for x in self.priorities}
        self.fix, self.made = {a: v for a, v in target}, {}
        return True

    def finish(self, query):
        """Shuffle the list orders and rename the ids, so that no position or id shows the role of an item."""
        rng = self.rng
        used = ({c["attr"] for x in self.rules + self.exceptions for c in x["if"]} |
                {r["then"]["attr"] for r in self.rules} | {a for _, a in self.facts} |
                set([query["attr"]] if query["type"] == "binary" else query["candidates"]))
        names = [a for a in self.attrs if a in used]
        new = dict(zip(names, ["a%d" % i for i in rng.sample(range(len(names)), len(names))]))
        cond = lambda c: {**c, "attr": new[c["attr"]]}
        rules = list(self.rules)
        rng.shuffle(rules)
        rid = {r["id"]: "r%d" % i for i, r in enumerate(rules)}
        exc = list(self.exceptions)
        rng.shuffle(exc)
        pri = list(self.priorities)
        rng.shuffle(pri)
        facts = [{"entity": e, "attr": new[b], "value": v} for (e, b), v in self.facts.items()]
        rng.shuffle(facts)
        theory = {
            "entities": list(self.ents),
            "attributes": {new[b]: self.attrs[b] for b in sorted(names, key=lambda b: int(new[b][1:]))},
            "facts": facts,
            "rules": [{"id": rid[r["id"]], "if": [cond(c) for c in r["if"]],
                       "then": {"attr": new[r["then"]["attr"]], "neg": r["then"]["neg"]}} for r in rules],
            "exceptions": [{"id": "x%d" % i, "if": [cond(c) for c in x["if"]], "blocks": rid[x["blocks"]]}
                           for i, x in enumerate(exc)],
            "priorities": [{"higher": rid[x["higher"]], "lower": rid[x["lower"]]} for x in pri],
        }
        if query["type"] == "binary":
            query = {**query, "attr": new[query["attr"]]}
        else:
            query = {**query, "candidates": [new[c] for c in query["candidates"]]}
        return theory, query


# ---------------------------------------------------------------- counterfactual edits (section 6)

def candidate_edits(t, etype, rng):
    """All edits of one type, in a random order: (edited theory, edit_path)."""
    out = []
    facts = t["facts"]
    if etype == "fact_flip":
        for i, f in enumerate(facts):
            if isinstance(f["value"], bool):
                out.append(({**t, "facts": facts[:i] + [{**f, "value": not f["value"]}] + facts[i + 1:]}, f"/facts/{i}/value"))
    elif etype == "fact_remove" and len(facts) > 1:
        for i in range(len(facts)):
            out.append(({**t, "facts": facts[:i] + facts[i + 1:]}, f"/facts/{i}"))
    elif etype == "fact_number":
        for i, f in enumerate(facts):
            if t["attributes"][f["attr"]] != "int":
                continue
            ks = [c["value"] for r in t["rules"] + t["exceptions"] for c in r["if"] if c["attr"] == f["attr"]]
            vals = {k + d for k in ks for d in (-1, 0, 1)} | {rng.randint(0, 99)}
            for v in sorted(w for w in vals if 0 <= w <= 99 and w != f["value"]):
                out.append(({**t, "facts": facts[:i] + [{**f, "value": v}] + facts[i + 1:]}, f"/facts/{i}/value"))
    elif etype == "threshold_shift":
        for key in ("rules", "exceptions"):
            for i, r in enumerate(t[key]):
                for j, c in enumerate(r["if"]):
                    if "op" not in c:
                        continue
                    vs = [f["value"] for f in facts if f["attr"] == c["attr"]]
                    ks = {v + d for v in vs for d in (-1, 0, 1)} | {rng.randint(0, 99)}
                    for k in sorted(w for w in ks if 0 <= w <= 99 and w != c["value"]):
                        r2 = {**r, "if": r["if"][:j] + [{**c, "value": k}] + r["if"][j + 1:]}
                        out.append(({**t, key: t[key][:i] + [r2] + t[key][i + 1:]}, f"/{key}/{i}/if/{j}/value"))
    elif etype == "exception_remove":
        for i in range(len(t["exceptions"])):
            out.append(({**t, "exceptions": t["exceptions"][:i] + t["exceptions"][i + 1:]}, f"/exceptions/{i}"))
    elif etype == "priority_swap":
        for i, x in enumerate(t["priorities"]):
            sw = {"higher": x["lower"], "lower": x["higher"]}
            out.append(({**t, "priorities": t["priorities"][:i] + [sw] + t["priorities"][i + 1:]}, f"/priorities/{i}"))
    elif etype == "rule_remove" and len(t["rules"]) > 1:  # T' keeps 1 rule or more
        for i in range(len(t["rules"])):
            out.append(({**t, "rules": t["rules"][:i] + t["rules"][i + 1:]}, f"/rules/{i}"))
    elif etype == "condition_negate":
        for i, r in enumerate(t["rules"]):
            for j, c in enumerate(r["if"]):
                if "neg" in c:
                    c2, key = {**c, "neg": not c["neg"]}, "neg"
                elif c["op"] in COMPLEMENT:
                    c2, key = {**c, "op": COMPLEMENT[c["op"]]}, "op"
                else:  # '==' has no complement in the operator set
                    continue
                r2 = {**r, "if": r["if"][:j] + [c2] + r["if"][j + 1:]}
                out.append(({**t, "rules": t["rules"][:i] + [r2] + t["rules"][i + 1:]}, f"/rules/{i}/if/{j}/{key}"))
    rng.shuffle(out)
    return out


def _shape(t, query):
    """Shallow structural counts of a theory (no fact lookup, no solving). `choose_edit` uses their change as the
    signature of an edit."""
    qa = [query["attr"]] if query["type"] == "binary" else list(query["candidates"])
    heads = defaultdict(list)
    for r in t["rules"]:
        heads[r["then"]["attr"]].append(r["then"]["neg"])
    reads, lits = Counter(), Counter()
    for x in t["rules"] + t["exceptions"]:
        for c in x["if"]:
            reads[c["attr"]] += 1
            lits["thr" if "op" in c else ("neg" if c["neg"] else "pos")] += 1
    qids = {r["id"] for r in t["rules"] if r["then"]["attr"] in qa}
    try:
        h = _heights(t, _check_theory(t))
    except ValueError:
        h = {}
    fact_attrs = {f["attr"] for f in t["facts"]}
    return (len(t["facts"]), sum(f["value"] is True for f in t["facts"]), sum(f["value"] is False for f in t["facts"]),
            len(t["rules"]), len(t["exceptions"]), len(t["priorities"]), len(heads),
            sum(all(v) for v in heads.values()), sum(not any(v) for v in heads.values()),
            sum(n == 1 for n in reads.values()), sum(reads[a] == 1 and a not in fact_attrs and a not in heads
                                                     for a in t["attributes"]),
            lits["pos"], lits["neg"], lits["thr"], max(h.values(), default=0), sum(h.get(a, 0) for a in qa),
            sum(not r["then"]["neg"] for r in t["rules"] if r["id"] in qids),
            sum(r["then"]["neg"] for r in t["rules"] if r["id"] in qids),
            sum(x["blocks"] in qids for x in t["exceptions"]), sum(p["higher"] in qids for p in t["priorities"]),
            sum(p["lower"] in qids for p in t["priorities"]), max((len(r["if"]) for r in t["rules"]), default=0),
            sum(len(r["if"]) for r in t["rules"])) + _qfacts(t, query["entity"])


def _qfacts(t, e):
    """The facts of entity e: true, false, and missing bool base facts, and int facts."""
    heads = {r["then"]["attr"] for r in t["rules"]}
    qf = {f["attr"]: f["value"] for f in t["facts"] if f["entity"] == e}
    b = [a for a, typ in t["attributes"].items() if typ == "bool" and a not in heads]
    return (sum(qf.get(a) is True for a in b), sum(qf.get(a) is False for a in b), sum(a not in qf for a in b),
            sum(t["attributes"][a] == "int" for a in qf))


# The shares of the free facts of the queried entity (`fill`): bool true, false, missing; int value, missing.
FACT_SHARES = {"bool": {True: 3 / 8, False: 3 / 8, None: 1 / 4}, "int": {"v": 3 / 4, None: 1 / 4}}

# Positions in `_shape` of the counts of the coarse key: facts, true and false facts, rules, exceptions, priorities,
# heads with only positive or only negative conclusions, the maximum height, the counts on the query attributes, and
# the facts of the queried entity (true, false, and missing bool facts, int facts).
COARSE = (0, 1, 2, 3, 4, 5, 7, 8, 14, 15, 16, 17, 18, 19, 20, 23, 24, 25, 26)


def _coarse(key):
    return ("coarse", key[0], key[1], tuple(key[2][i] for i in COARSE))


def choose_edit(t, query, orig, want_type, want_flip, feats, rng, want_label=None, quota=None, ratio=1.5,
                inert_ok=False):
    """Choose the edit (section 9.2). Solve every candidate edit of every permitted type. Keep a flip, or a stay that
    changes a derived value of the query entity (a stay that changes nothing is inert). For binary depth 0 a stay
    that changes a fact of the query entity counts. `relevant` (meta edit_relevant) is stricter: the change is on the path of the query (`_cone`).
    The key of an edit is (task, edit type, signature), where the signature is the change of the shallow counts
    (`_shape`). quota[key] = [stays, flips] of the accepted units so far. The room of a flip is ratio * stays - flips,
    and the room of a stay is flips / ratio - stays, the smaller of the values for the key and for its coarse key
    (`_coarse`: the structural counts only). The edit with the most room wins, so that each key moves to
    flips : stays = ratio : 1 and the shallow counts of the pair do not show the flip status. The room is a preference,
    not a limit: a key that has few stays (or few flips) in the theories does not block a unit.
    Take the planned edit type if it has a flip (to want_label, if set) AND a stay (else the first other type that has
    both, the type with the fewest accepted units first). This test is the same for a flip and a stay plan, so flips
    and stays come from the same theories and the same edit types. The test does not use the quota: if it did, a key
    with few stays would also block the stays that it needs. Then take the edit of the planned flip status with the
    most room. A binary depth-0 Unknown has no flip (no edit adds a fact), so it takes the fact_flip or fact_remove
    stay with the most room (the types of depth-0 flips), without the type test. If inert_ok (the inert budget of the
    run is not used up), a type without a stay that changes a value can use an inert stay (not at binary depth 0).
    Return (edit_type, edited theory, edit_path, solve result, relevant, key, inert) or None."""
    quota = {} if quota is None else quota
    cone = _cone(t, query)
    before = _solve(t, query)[1]
    e = query["entity"]
    heads = {r["then"]["attr"] for r in t["rules"]}
    base = _shape(t, query)
    any_stay = query["type"] == "binary" and orig["depth"] == 0
    no_flip = any_stay and orig["label"] == "Unknown"
    qf0 = sorted((f["attr"], f["value"]) for f in t["facts"] if f["entity"] == e)
    cands = defaultdict(lambda: ([], []))  # edit type -> (open stays, open flips), each item (room, edit)
    inert = defaultdict(list)  # edit type -> open inert stays
    for et in permitted_edits(feats):
        for t2, path in candidate_edits(t, et, rng):
            try:
                res, val = _solve(t2, query)
            except ValueError:  # T' is not valid: skip the candidate (section 6)
                continue
            flip = res["label"] != orig["label"]
            rel = flip or any(before.get((e, a)) != val.get((e, a)) for a in cone)
            touch = rel or any(before.get((e, a)) != val.get((e, a)) for a in heads)
            # spec: section 9.2. At depth 0 a stay changes a fact of q: a flip always does, so a stay on another
            # entity would show the stay.
            touch = touch or (any_stay and sorted((f["attr"], f["value"]) for f in t2["facts"] if f["entity"] == e) != qf0)
            if not (touch or (inert_ok and not any_stay)):
                continue
            sig = tuple(y - x for x, y in zip(base, _shape(t2, query)))
            key = (query["type"], et, sig)
            room = min(ratio * n_s - n_f if flip else n_f / ratio - n_s
                       for n_s, n_f in (quota.get(k, (0, 0)) for k in (key, _coarse(key))))
            (cands[et][flip] if touch else inert[et]).append((room, (et, t2, path, res, rel, key, not touch)))
    if no_flip:  # spec: section 9.2. Its stays go to the keys of all types that need stays most. With the type test,
        # the rarest type with a stay took most of them, and its flip share fell to 0.36.
        # Only the edit types of depth-0 flips: a rule edit at depth 0 is always a stay.
        pick = [x for et in ("fact_flip", "fact_remove") for x in cands[et][0]]
        top = max((x[0] for x in pick), default=None)
        return None if top is None else rng.choice([x[1] for x in pick if x[0] == top])
    others = [x for x in permitted_edits(feats) if x != want_type]
    rng.shuffle(others)
    others.sort(key=lambda x: quota.get(("type", x), 0))  # the rarest edit type first (a stable sort keeps the ties random)
    for et in [want_type] + others:
        stays, flips = cands[et]
        stays = stays or inert[et]
        if want_label is not None:
            flips = [x for x in flips if x[1][3]["label"] == want_label]
        if not stays or not flips:
            continue
        pick = flips if want_flip and flips else stays
        top = max(x[0] for x in pick)  # ponytail: a preference, not a limit (a limit blocks the units of rare keys)
        return rng.choice([x[1] for x in pick if x[0] == top])
    return None


# ---------------------------------------------------------------- rendering (section 5.2)

def make_lexicon(theory, pool_id, rng):
    pool = VOCAB[pool_id]
    attrs = theory["attributes"]
    bools = [a for a in attrs if attrs[a] == "bool"]
    ints = [a for a in attrs if attrs[a] == "int"]
    lex = dict(zip(theory["entities"], rng.sample(pool["names"], len(theory["entities"]))))
    lex.update(zip(bools, rng.sample(pool["bool"], len(bools))))
    lex.update(zip(ints, rng.sample(pool["int"], len(ints))))
    return lex


def render(theory, query, lex, pool_id, tid):
    """Return (state, instruction, options). A pure function of its arguments."""
    fam, pool = FAMILIES[tid], VOCAB[pool_id]
    kind, frames, attrs = pool["kind"], pool["frames"], theory["attributes"]

    def fact(f):
        if attrs[f["attr"]] == "int":
            return frames["int"].format(lex[f["entity"]], lex[f["attr"]]["fact"].format(f["value"]))
        return frames["bool_pos" if f["value"] else "bool_neg"].format(lex[f["entity"]], lex[f["attr"]][0 if f["value"] else 1])

    def cond(c, first):
        # spec: section 5.2. Only the first condition has the condition frame (with its subject). The other
        # conditions are the bare pool phrase, so that the subject is not repeated ("if a lender is old and has ...").
        frame = fam["cond"] if first else "{x}"
        if "op" in c:
            x = lex[c["attr"]]["thr"].format(fam["ops"][c["op"]].format(k=c["value"]))
            return frame.format(x=x, kind=kind)
        return frame.format(x=lex[c["attr"]][1 if c["neg"] else 0], kind=kind)

    pos = {r["id"]: i for i, r in enumerate(theory["rules"])}
    lab = lambda rid: rule_label(tid, pos[rid])
    conds = lambda cs: fam["join"].join(cond(c, i == 0) for i, c in enumerate(cs))
    parts = []
    for r in theory["rules"]:
        th = r["then"]
        concl = fam["concl"].format(x=lex[th["attr"]][1 if th["neg"] else 0], kind=kind)
        parts.append(fam["rule"].format(L=lab(r["id"]), kind=kind, C=conds(r["if"]), K=concl))
    parts += [fam["exc"].format(L=lab(x["blocks"]), kind=kind, C=conds(x["if"])) for x in theory["exceptions"]]
    parts += [fam["pri"].format(H=lab(x["higher"]), Lo=lab(x["lower"])) for x in theory["priorities"]]
    name = lex[query["entity"]]
    if query["type"] == "binary":
        question = fam["q"].format(name=name, x=lex[query["attr"]][1 if query["neg"] else 0])
        options = list(LABELS)
    else:
        question = fam["q_mc"].format(name=name, kind=kind)
        options = [lex[c][0] for c in query["candidates"]]
    state = " ".join(fact(f) for f in theory["facts"])
    return state, " ".join(parts + ["Question: " + question]), options


# ---------------------------------------------------------------- generation (sections 8 and 9)

def generate(config, seed, stats=None):
    """Yield (theory, query, record) for each accepted record, in output order (section 5.4).
    Call the engine for each record (section 9.3). If stats is a dict, add the counters of section 9.3 to it."""
    engine = None
    # spec: section 10.2 says the engine import is inside generate. `engine_check: false` (tests only)
    # skips the engine, so that the generator tests do not depend on the engine code.
    if config.get("engine_check", True):
        from crux.gen import engine
    stats = {} if stats is None else stats
    for key in ("units_attempted", "units_accepted", "unit_failed", "attempts", "redrawn"):
        stats.setdefault(key, 0)
    rejections = stats.setdefault("rejections", Counter())
    examples = stats.setdefault("disagreements", {})
    cells = stats.setdefault("cells", {})  # (composition, depth, task) -> [planned units, accepted units]
    total, caps = config["total_records"], config["caps"]
    seen = set()
    flips = defaultdict(lambda: [0, 0])  # composition -> [accepted units, flips]
    quota = {}  # edit key -> [accepted stays, accepted flips] (choose_edit)
    labels = defaultdict(Counter)  # composition -> label of the binary orig records at depth >= 1 -> count
    dest = defaultdict(Counter)  # composition -> label of the edit member of the binary flips -> count
    positions = {n: [0] * n for n in range(config["option_counts"][0], config["option_counts"][1] + 1)}
    produced, k = 0, 0
    while total - produced >= 2:
        # spec: section 9.2. The plan probability of a flip follows the flip deficit of the composition, so that
        # each composition reaches the flip target (a composition can permit fewer flips than another).
        plan0 = make_plan(config, seed, k)
        n_c, f_c = flips[plan0["composition"]]
        flip_p = min(1.0, max(0.0, config["flip_share"] + config["flip_gain"] * (config["flip_share"] * n_c - f_c)))
        # spec: section 9.2. The orig label weights follow the label deficit of the binary orig records of the
        # composition. A global controller moves labels between compositions, and the counts show the composition.
        # Depth 0 has uniform weights and is not counted: a depth-0 flip cannot be Unknown, so a controller would
        # give the deeper units of the composition more Unknown labels, and the depth would show the label.
        lc = labels[plan0["composition"]]
        label_w = None if plan0["depth"] == 0 else \
            [max(0.02, 1 / 3 + config["label_gain"] * (sum(lc.values()) / 3 - lc[x])) for x in LABELS]
        pos_w = {n: [max(0.02, 1 / n + config["position_gain"] * (sum(c) / n - x)) for x in c] for n, c in positions.items()}
        # spec: section 9.2. The edit label weights follow the deficit of the labels of the binary flips (edit
        # members) of the composition, so that the flips do not end in Unknown more often than in True or False.
        # Depth 0: the orig labels have few Unknown (a depth-0 flip cannot be Unknown), so a depth-0 flip ends in
        # Unknown with weight 2/3 and all depth-0 records have about 1/3 of each label.
        dc = dest[plan0["composition"]]
        edit_w = {"True": 1 / 3, "False": 1 / 3, "Unknown": 2 / 3} if plan0["depth"] == 0 else \
            {x: max(0.02, 1 / 3 + config["label_gain"] * (sum(dc.values()) / 3 - dc[x])) for x in LABELS}
        # spec: section 9.2. The planned edit type has the weight (mean accepted units of a type) / (accepted units
        # of the type), at most 4. The fallback order ends in the types that most theories permit, so the other
        # types need more plans to reach their share.
        acc = [quota.get(("type", t), 0) for t in EDIT_TYPES]
        type_w = {t: min(4.0, (sum(acc) / len(acc) + 1) / (a + 1)) for t, a in zip(EDIT_TYPES, acc)}
        plan = make_plan(config, seed, k, flip_p, label_w, pos_w, edit_w, type_w)
        unit_id = "u%06d" % k
        stats["units_attempted"] += 1
        cell = cells.setdefault("%s/%d/%s" % (plan["composition"], plan["depth"], plan["task"]), [0, 0])
        cell[0] += 1
        templates = plan["templates"][:(total - produced) // 2]
        long = 0
        for j in range(config["max_attempts"]):
            stats["attempts"] += 1
            rng = random.Random(derive_seed(seed, "attempt", k, j))
            # spec: section 9.3. The word budget: after every 2 attempts that break a length cap, the unit uses
            # fewer optional parts (decoys become fillers, fewer noise facts and extra conditions).
            unit, reason = _attempt(plan, rng, j, config, templates, unit_id, caps, seen, long // 2, quota), None
            if isinstance(unit, str):
                reason, unit = unit, None
            elif engine is not None:
                reason = _engine_check(engine, unit, examples)
            if reason:
                rejections[reason] += 1
                long += reason == "too_long"
                continue
            stats["redrawn"] += unit["redrawn"]
            m0 = unit["records"][0][2]["meta"]
            for qk in (unit["key"], _coarse(unit["key"])):
                quota.setdefault(qk, [0, 0])[m0["label_flip"]] += 1
            quota[("type", m0["edit_type"])] = quota.get(("type", m0["edit_type"]), 0) + 1
            if not m0["label_flip"] and not (plan["task"] == "binary" and plan["depth"] == 0):
                n_st = quota.setdefault("inert", [0, 0])
                n_st[0] += 1
                n_st[1] += unit["inert"]
            stats["units_accepted"] += 1
            lean = stats.setdefault("lean", {}).setdefault("%s/%d" % (plan["task"], plan["depth"]), [0, 0, 0, 0])
            lean[unit["lean"]] += 1
            cell[1] += 1
            flips[plan["composition"]][0] += 1
            flips[plan["composition"]][1] += unit["records"][0][2]["meta"]["label_flip"]
            if plan["task"] == "binary" and m0["label_flip"] and plan["depth"] >= 1:
                dest[plan["composition"]][unit["members"][1][2]["label"]] += 1
            for (theory, query, rec) in unit["records"]:
                if rec["task_type"] == "binary":
                    labels[plan["composition"]][rec["meta"]["label"]] += rec["meta"]["pair_role"] == "orig" and \
                        plan["depth"] >= 1
                else:
                    positions[len(rec["options"])][rec["target"]] += 1
                seen.add((rec["state"], rec["instruction"]))
                seen.add((rec["meta"]["theory_hash"], canonical(query)))
                produced += 1
                yield theory, query, rec
            break
        else:
            stats["unit_failed"] += 1
        k += 1


def _build_and_edit(plan, rng, cfg, lean, quota=None, late=False):
    """Build the theory and choose the edit. Return (theory, query, solve result, edit, lean), or "plan_miss".
    `late`: the second half of the attempts of a unit, where an inert stay is permitted."""
    lean = min(lean, 3)
    built = _Builder(plan, rng, cfg, lean).build()
    while built == "long" and lean < 3:  # spec: section 5.2, the word budget: build again with fewer optional parts
        lean += 1
        built = _Builder(plan, rng, cfg, lean).build()
    if built is None or built == "long":
        return "plan_miss"
    t, query = built
    try:
        mine = solve(t, query)
    except ValueError:
        return "plan_miss"
    want = query["candidates"][plan["label"]] if plan["task"] == "multi_choice" else plan["label"]
    if mine["label"] != want or mine["depth"] != plan["depth"] or features(t) != plan["features"]:
        return "plan_miss"
    # spec: section 9.2. The inert cap does not change during the attempts of one unit, so a unit without a relevant
    # stay would fail all attempts. Thus the second half of the attempts permits an inert stay.
    stays, inert = (quota or {}).get("inert", (0, 0))
    edit = choose_edit(t, query, mine, plan["edit_type"], plan["flip"], plan["features"], rng, plan.get("edit_label"),
                       quota, cfg["flip_share"] / (1 - cfg["flip_share"]),
                       late or inert < cfg["inert_share"] * (stays + 1))
    return "no_balanced_edit" if edit is None else (t, query, mine, edit, lean)


def _attempt(plan, rng, j, cfg, templates, unit_id, caps, seen, lean=0, quota=None):
    """One attempt of one unit: build, edit, render. Return the unit, or the rejection reason (str).
    The flip status follows the plan (only a binary depth-0 Unknown takes a stay for a flip plan)."""
    got = _build_and_edit(plan, rng, cfg, lean, quota, j >= cfg["max_attempts"] // 2)
    if isinstance(got, str):
        return got
    t, query, mine, (et, t2, path, mine2, relevant, ekey, inert), lean = got
    mc = plan["task"] == "multi_choice"
    flip = mine2["label"] != mine["label"]
    lex = make_lexicon(t, plan["vocab"], rng)
    records, texts = [], set()
    members = (("orig", t, mine), ("edit", t2, mine2))
    hashes = {role: theory_hash(th) for role, th, _ in members}
    for tid in templates:
        for role, th, res in members:
            state, instruction, options = render(th, query, lex, plan["vocab"], tid)
            target = query["candidates"].index(res["label"]) if mc else options.index(res["label"])
            meta = {"template_id": tid, "template_group": plan["template_group"], "composition_id": plan["composition"],
                    "vocabulary_id": plan["vocab"], "block": plan["block"], "depth": res["depth"],
                    "unit_depth": mine["depth"], "label": options[target], "unit_id": unit_id,
                    "pair_id": f"{unit_id}-{tid}", "pair_role": role, "edit_type": et, "edit_path": path,
                    "label_flip": flip, "edit_relevant": relevant, "paraphrase_group": f"{unit_id}-{role}",
                    "rule_types": features(th),
                    "theory_hash": hashes[role], "query": query, "attempt": j}
            records.append((th, query, {"id": f"syn-{unit_id}-{tid}-{role}", "source": "synthetic", "task_type": plan["task"],
                                        "state": state, "instruction": instruction, "options": options,
                                        "target": target, "meta": meta}))
    recs = [r for _, _, r in records]
    if any(len(r["state"].split()) > caps["state_words"] or len(r["instruction"].split()) > caps["instruction_words"]
           for r in recs):
        return "too_long"
    for r in recs:
        key = (r["state"], r["instruction"])
        if key in seen or key in texts:
            return "duplicate"
        texts.add(key)
    for role, th, _ in members:  # the same theory and query in two units is a duplicate too
        if (hashes[role], canonical(query)) in seen:
            return "duplicate"
    for role in ("orig", "edit"):  # section 7: the paraphrase group check
        group = [r for r in recs if r["meta"]["pair_role"] == role]
        same = {json.dumps([r["meta"]["theory_hash"], r["meta"]["query"], r["state"], r["options"], r["target"]])
                for r in group}
        if len(same) != 1 or len({r["instruction"] for r in group}) != len(group):
            raise RuntimeError(f"bad paraphrase group {unit_id}-{role}")
    redrawn = (et, flip) != (plan["edit_type"], plan["flip"])
    return {"records": records, "members": members, "query": query, "redrawn": int(redrawn), "lean": lean, "key": ekey,
            "inert": int(inert)}


def _engine_check(engine, unit, examples):
    """The engine checks of section 9.3, in order. Return the rejection reason or None."""
    query = unit["query"]
    for _, th, _ in unit["members"]:
        errors = engine.validate(th, query)
        if errors:
            return "invalid:" + errors[0].split(":")[0]
    results = []
    for _, th, res in unit["members"]:
        try:
            results.append((th, res, engine.evaluate(th, query)))
        except ValueError as e:  # validate() gave no error, but evaluate() refuses: count it as a label disagreement
            results.append((th, res, {"label": f"ValueError: {e}", "depth": None}))
    for reason, get in (("disagree_label", lambda th, res, ev: (ev["label"], res["label"])),
                        ("disagree_depth", lambda th, res, ev: (ev["depth"], res["depth"])),
                        ("disagree_features", lambda th, res, ev: (engine.features(th), features(th)))):
        for th, res, ev in results:
            theirs, ours = get(th, res, ev)
            if theirs != ours:
                ex = examples.setdefault(reason, [])
                if len(ex) < 5:
                    ex.append({"theory": th, "query": query, "engine": theirs, "generator": ours})
                return reason
    return None


# ---------------------------------------------------------------- outputs and report (sections 5.4 and 9.4)

def _share(counter, keys):
    n = sum(counter.values())
    return {str(k): round(counter[k] / n, 4) if n else None for k in keys}


def _targets(records, cfg):
    """The realized value of every target of section 9.2, and the names of the missed targets."""
    units = {}
    for r in records:
        m = r["meta"]
        u = units.setdefault(m["unit_id"], {"block": m["block"], "task": r["task_type"], "n": len(r["options"]),
                                            "depth": m["unit_depth"], "comp": m["composition_id"], "records": 0,
                                            "neg": m["query"].get("neg")})
        u["records"] += 1
    out, miss = {}, []

    def check(name, value, ok):
        out[name] = value
        if not ok:
            miss.append(name)

    blocks = ("train", "val", "test")
    bs = _share(Counter(u["block"] for u in units.values()), blocks)
    check("block_shares", bs, all(bs[b] is not None and abs(bs[b] - cfg["block_shares"][b]) <= 0.01 for b in blocks))
    for b in blocks:
        bu = [u for u in units.values() if u["block"] == b]
        s = round(sum(u["task"] == "multi_choice" for u in bu) / len(bu), 4) if bu else None
        check(f"multi_choice_share/{b}", s, s is not None and abs(s - cfg["multi_choice_share"]) <= 0.02)
    lo, hi = cfg["option_counts"]
    # spec: section 9.2. The option count is uniform from lo to max_options of the unit (composition kind, depth),
    # so the target is the mean of these uniform shares over the multi-choice units.
    mcu = [u for u in units.values() if u["task"] == "multi_choice"]
    want = Counter()
    for u in mcu:
        top = cfg["max_options"]["conj" if "conj" in comp_features(u["comp"]) else "single"][u["depth"]]
        for n in range(lo, top + 1):
            want[n] += 1 / (top - lo + 1) / len(mcu)
    oc = _share(Counter(u["n"] for u in mcu), range(lo, hi + 1))
    check("option_counts", {"realized": oc, "target": {str(n): round(want[n], 4) for n in range(lo, hi + 1)}},
          all(v is not None and abs(v - want[int(n)]) <= 0.02 for n, v in oc.items()))
    # spec: section 9.2. The orig labels at depth 1 or more are balanced per composition. A depth-0 Unknown cannot
    # flip, so depth 0 has fewer Unknown labels, and 7 compositions have no True <-> False flip, so their edits have
    # more. Thus all records have a wider tolerance.
    for b in blocks:
        for name, roles, d0, tol in (("binary_orig_labels", ("orig",), 1, 0.03 if b == "train" else 0.05),
                                     ("binary_labels", ("orig", "edit"), 0, 0.05)):
            lab = _share(Counter(r["meta"]["label"] for r in records if r["meta"]["block"] == b and
                                 r["task_type"] == "binary" and r["meta"]["pair_role"] in roles and
                                 r["meta"]["unit_depth"] >= d0), LABELS)
            check(f"{name}/{b}", lab, all(v is not None and abs(v - 1 / 3) <= tol for v in lab.values()))
    # Not a target: the binary labels for each pair role, so that a skew of the edit labels (most flips end in
    # Unknown) stays visible when the totals are balanced.
    role = lambda r: r["meta"]["pair_role"] + ("" if r["meta"]["pair_role"] == "orig" else
                                               "_flip" if r["meta"]["label_flip"] else "_stay")
    out["binary_labels_by_role"] = {k: _share(Counter(r["meta"]["label"] for r in records
                                                      if r["task_type"] == "binary" and role(r) == k), LABELS)
                                    for k in ("orig", "edit_flip", "edit_stay")}
    for n in range(lo, hi + 1):
        pos = _share(Counter(r["target"] for r in records if r["task_type"] == "multi_choice" and len(r["options"]) == n), range(n))
        check(f"mc_target_position/{n}", pos, all(v is not None and abs(v - 1 / n) <= 0.03 for v in pos.values()))
    for b in blocks:
        w = cfg["depth_weights"][b]
        dep = _share(Counter(u["depth"] for u in units.values() if u["block"] == b), range(len(w)))
        tol = 0.02 if b == "train" else 0.04
        check(f"depth/{b}", dep, all(dep[str(d)] is not None and abs(dep[str(d)] - w[d] / sum(w)) <= tol for d in range(len(w)))
              and all(u["depth"] < len(w) for u in units.values() if u["block"] == b))
    for b in blocks:
        c = Counter(u["comp"] for u in units.values() if u["block"] == b)
        n = sum(c.values())
        check(f"compositions/{b}", {x: c[x] for x in COMPOSITIONS[b]},
              n > 0 and all(c[x] >= 0.5 * n / len(COMPOSITIONS[b]) for x in COMPOSITIONS[b]))
    tu = [u for u in units.values() if u["block"] == "train"]
    cov = {f: round(sum(f in comp_features(u["comp"]) for u in tu) / len(tu), 4) if tu else None for f in FEATURES}
    check("rule_type_coverage/train", cov, all(v is not None and v >= 0.40 for v in cov.values()))
    pairs = [r for r in records if r["meta"]["pair_role"] == "orig"]
    fl = round(sum(r["meta"]["label_flip"] for r in pairs) / len(pairs), 4) if pairs else None
    check("label_flips", fl, fl is not None and abs(fl - cfg["flip_share"]) <= 0.05)
    for task in ("binary", "multi_choice"):
        tp = [r for r in pairs if r["task_type"] == task]
        s = round(sum(r["meta"]["label_flip"] for r in tp) / len(tp), 4) if tp else None
        check(f"label_flips/{task}", s, s is not None and s >= 0.40)
    for et in EDIT_TYPES:
        ep = [r for r in pairs if r["meta"]["edit_type"] == et]
        share = round(len(ep) / len(pairs), 4) if pairs else None
        fs = round(sum(r["meta"]["label_flip"] for r in ep) / len(ep), 4) if ep else None
        check(f"edit_types/{et}", {"share": share, "flip_share": fs},
              share is not None and share >= (0.005 if et in ("exception_remove", "priority_swap") else 0.05)
              and fs is not None and 0.10 <= fs <= 0.90)
    tpu = _share(Counter(u["records"] // 2 for u in units.values()), (1, 2, 3))
    check("templates_per_unit", tpu, all(v is not None and abs(v - w) <= 0.02 for v, w in zip(tpu.values(), cfg["templates_per_unit"])))
    bn = [u for u in units.values() if u["task"] == "binary"]
    out["binary_neg_share"] = round(sum(u["neg"] for u in bn) / len(bn), 4) if bn else None
    return out, miss


def _effective_options(records, theories):
    """Mean option count of the multi-choice records, and the mean count of options that a reader cannot reject
    from the surface: the candidate is the conclusion of a positive rule, or a true fact of the query entity."""
    n, live = [], []
    for r in records:
        if r["task_type"] != "multi_choice":
            continue
        t, q = theories[r["meta"]["theory_hash"]], r["meta"]["query"]
        ok = {x["then"]["attr"] for x in t["rules"] if not x["then"]["neg"]}
        ok |= {f["attr"] for f in t["facts"] if f["entity"] == q["entity"] and f["value"] is True}
        n.append(len(q["candidates"]))
        live.append(sum(c in ok for c in q["candidates"]))
    return {"options": round(sum(n) / len(n), 3), "effective": round(sum(live) / len(live), 3)} if n else None


def tokenizer_check(records, seed):
    """Count the truncated records in a sample of 2,000 (section 5.2). Return None if the tokenizer is not local."""
    try:
        from transformers import AutoTokenizer

        from crux.data.batching import EncodedDataset
        cfg = yaml.safe_load(open(ROOT / "configs" / "step2_data.yaml"))
        tok = AutoTokenizer.from_pretrained(cfg["tokenizer"], local_files_only=True)
    except Exception:  # no local tokenizer: the check does not run
        return None
    sample = random.Random(derive_seed(seed, "tokenizer")).sample(records, min(2000, len(records)))
    return {"sample": len(sample),
            "joint_options": EncodedDataset(sample, tok, cfg, "joint", joint_options=True).truncated["records"],
            "separate": EncodedDataset(sample, tok, cfg, "separate").truncated["records"]}


def run(config, seed, out_dir):
    """Generate, write all.jsonl, theories.jsonl, and report.json. Return the report."""
    stats, records, theories = {}, [], {}
    for theory, _, rec in generate(config, seed, stats):
        records.append(rec)
        theories.setdefault(rec["meta"]["theory_hash"], theory)
    out = Path(out_dir)
    write_jsonl(records, out / "all.jsonl")
    with open(out / "theories.jsonl", "w") as f:
        for h, t in theories.items():
            f.write(json.dumps({"theory": t, "theory_hash": h}, sort_keys=True, ensure_ascii=False) + "\n")
    targets, misses = _targets(records, config)
    disagree = sum(v for k, v in stats["rejections"].items() if k.startswith("disagree_"))
    count = lambda key: dict(sorted(Counter(r["meta"][key] for r in records).items()))
    report = {
        "config": {k: v for k, v in config.items() if k != "out_dir"},
        "seed": seed,
        "python": "%d.%d" % sys.version_info[:2],  # spec: the minor version only (section 8), so the bytes match
        "counts": {"units_attempted": stats["units_attempted"], "units_accepted": stats["units_accepted"],
                   "unit_failed": stats["unit_failed"], "attempts": stats["attempts"],
                   "redrawn_units": stats["redrawn"],
                   "records": len(records), "theories": len(theories)},
        "rejections": dict(sorted(stats["rejections"].items())),
        "disagree_share": round(disagree / stats["attempts"], 6) if stats["attempts"] else 0.0,
        "disagreements": stats["disagreements"],
        "targets": targets,
        "missed_targets": misses,
        # accepted / planned units for each cell (composition/depth/task): a failed unit is a gap in its cell
        "fill_rate": {c: round(a / n, 4) for c, (n, a) in sorted(stats["cells"].items())},
        "by_composition_id": count("composition_id"), "by_template_id": count("template_id"),
        "by_vocabulary_id": count("vocabulary_id"), "by_edit_type": count("edit_type"), "by_depth": count("depth"),
        "mc_effective_options": _effective_options(records, theories),
        "by_unit_depth": dict(sorted(Counter({r["meta"]["unit_id"]: r["meta"]["unit_depth"] for r in records}.values()).items())),
        "truncated": tokenizer_check(records, seed) if config.get("tokenizer_check", True) else None,
        "sha256": {name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in ("all.jsonl", "theories.jsonl")},
    }
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    return report


def main():
    ap = argparse.ArgumentParser(description="Generate the Step 3 synthetic data.")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--total", type=int)
    ap.add_argument("--out")
    args = ap.parse_args()
    config = yaml.safe_load(open(args.config))
    if args.total:
        config["total_records"] = args.total
    seed = config["seed"] if args.seed is None else args.seed
    config["seed"] = seed
    out = Path(args.out or ROOT / config["out_dir"])
    start = time.time()
    report = run(config, seed, out)
    secs = time.time() - start
    print(json.dumps({k: report[k] for k in ("counts", "rejections", "missed_targets", "truncated")}, indent=1))
    print(f"{report['counts']['records']} records in {secs:.1f} s ({report['counts']['records'] / secs:.0f} records/s) -> {out}")
    errors = []
    if report["missed_targets"]:
        errors.append(f"missed targets: {report['missed_targets']}")
    if report["disagree_share"] > config["max_disagree_share"]:
        errors.append(f"engine disagreements in {report['disagree_share']:.4%} of the attempts")
    if errors:
        raise SystemExit("FAILED: " + "; ".join(errors))


if __name__ == "__main__":
    main()
