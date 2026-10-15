"""Step 3: the 15 template families t00 to t14 (docs/generator_spec.md, section 4.4).

A family renders only the instruction. Placeholders:
  label: {n} number, {A} letter, {R} roman numeral (rule position i gives n = i + 1).
  rule: {L} label, {kind} entity noun, {C} joined conditions, {K} conclusion.
  cond (bool positive, bool negative, threshold) and concl: {x} pool phrase, {kind}.
  ops: {k} threshold value. The pool threshold frame takes the op phrase.
  exc: {L} label of the blocked rule, {kind}, {C}. pri: {H} higher label, {Lo} lower label.
  q (binary, positive and negative literal): {name}, {x}. mc: {name}, {kind}.
Group of family t{n:02d}: "tg%d" % (n // 3).
"""

OPS = (">=", ">", "<=", "<", "==")

_ROWS = [
    # label, rule, join, cond, concl, exc, pri, q, mc, ops (>=, >, <=, <, ==)
    ("({n})", "{L} If a {kind} {C}, then {K}.", " and ", "{x}", "it {x}",
     "{L} is void if the {kind} {C}.", "When {H} and {Lo} clash, {H} prevails.",
     "Is it true that {name} {x}?", "Which option holds for {name}?",
     ("of at least {k}", "above {k}", "of at most {k}", "below {k}", "of exactly {k}")),
    ("Clause {n}", "{L}: once {C}, {K}.", " and also ", "any {kind} {x}", "it also {x}",
     "{L} does not hold for a {kind} when {C}.", "{H} overrides {Lo}.",
     "Does it follow that {name} {x}?", "Which of these is true of {name}?",
     ("of {k} or more", "over {k}", "of {k} or less", "under {k}", "equal to {k}")),
    ("Item {A}", "{L}. When {C}, {K}.", ", and ", "the {kind} {x}", "that {kind} {x}",
     "{L} is suspended when {C}.", "{H} outranks {Lo}.",
     "Can one infer that {name} {x}?", "What applies to {name}?",
     ("of no less than {k}", "of more than {k}", "of no more than {k}", "of less than {k}", "of precisely {k}")),
    ("Section {n}", "{L}: a {kind} {K} provided that {C}.", " as well as ", "it {x}", "{x}",
     "{L} lapses for a {kind} if {C}.", "{H} takes precedence over {Lo}.",
     "{name} {x}. Correct, wrong, or open?", "Which statement fits {name}?",
     ("of {k} or higher", "exceeding {k}", "of {k} or lower", "lower than {k}", "that equals {k}")),
    ("Policy {R}", "{L}. Suppose {C}: {K}.", " plus ", "that {kind} {x}", "then it {x}",
     "{L} is waived if {C}.", "In a clash, {H} beats {Lo}.",
     "Is it so that {name} {x}?", "Which choice is right for {name}?",
     ("no lower than {k}", "higher than {k}", "no higher than {k}", "smaller than {k}", "of {k} exactly")),
    ("#{n}", "{L} Each time {C}, {K}.", " and additionally ", "a {kind} {x}", "it then {x}",
     "{L} is set aside if {C}.", "{Lo} yields to {H}.",
     "Do the terms above show that {name} {x}?", "Which of the listed options is correct for {name}?",
     ("of {k} or above", "greater than {k}", "of {k} or below", "less than {k}", "matching {k}")),
    ("Provision {n}", "{L}: when {C}, {K}.", " while ", "the {kind} here {x}", "thus it {x}",
     "{L} has no effect when {C}.", "{H} has priority over {Lo}.",
     "Would it be right to say that {name} {x}?", "Which one describes {name}?",
     ("at or above {k}", "beyond {k}", "at or below {k}", "beneath {k}", "identical to {k}")),
    ("Point {A}", "{L}. If {C}, {K}.", " and further ", "said {kind} {x}", "it thereby {x}",
     "{L} is cancelled if {C}.", "{H} supersedes {Lo}.",
     "Statement: {name} {x}. Does it hold?", "Which choice is true for {name}?",
     ("of minimum {k}", "in excess of {k}", "of maximum {k}", "short of {k}", "set at {k}")),
    ("No. {n}", "{L} While {C}, {K}.", " and likewise ", "the given {kind} {x}", "so it {x}",
     "{L} stops applying as long as {C}.", "{H} wins over {Lo}.",
     "Is the claim \"{name} {x}\" true, false, or unknown?", "Select what holds for {name}: which option?",
     ("not below {k}", "that exceeds {k}", "not above {k}", "that falls below {k}", "that is exactly {k}")),
    ("Article {n}", "{L}: if {C}, {K}.", " and moreover ", "the subject {x}", "it therefore {x}",
     "{L} is inactive if {C}.", "{H} trumps {Lo}.",
     "Given the above, is it the case that {name} {x}?", "What can be said of {name}?",
     ("equal to or above {k}", "strictly above {k}", "equal to or below {k}", "strictly below {k}", "of {k} precisely")),
    ("Guideline {A}", "{L} - in case {C}, {K}.", "; and ", "one {kind} {x}", "it hence {x}",
     "{L} is disregarded in case {C}.", "{H} comes before {Lo}.",
     "Check whether {name} {x}: true, false, or undetermined?", "Which of the following applies to {name}?",
     ("of {k} or greater", "larger than {k}", "of {k} or smaller", "of below {k}", "exactly equal to {k}")),
    ("§{n}", "{L} If {C}, then {K}.", " and in addition ", "the {kind} concerned {x}", "this {kind} {x}",
     "{L} is excluded if {C}.", "{H} dominates {Lo}.",
     "Is it established that {name} {x}?", "Which option is established for {name}?",
     ("that reaches {k}", "that tops {k}", "that stays within {k}", "that stays under {k}", "fixed at {k}")),
    ("Term {n}", "{L}. Provided {C}, {K}.", " and besides ", "this one {x}", "it thus {x}",
     "{L} is withdrawn provided {C}.", "{Lo} gives way to {H}.",
     "What is the truth value of the statement that {name} {x}?", "Which description matches {name}?",
     ("of {k} at minimum", "surpassing {k}", "of {k} at maximum", "not reaching {k}", "that matches {k}")),
    ("¶{n}", "{L}: given that {C}, {K}.", " and simultaneously ", "that one {x}", "hence it {x}",
     "{L} is blocked given that {C}.", "{H} prevails over {Lo}.",
     "Can we confirm that {name} {x}?", "Which of these does {name} satisfy?",
     ("that is {k} or more", "that is over {k}", "that is {k} or less", "that is under {k}", "precisely equal to {k}")),
    ("Directive {n}", "{L}. Where {C}, {K}.", " and equally ", "the same {x}",
     "this {kind} then {x}",
     "{L} does not operate where {C}.", "Should {H} and {Lo} conflict, {H} applies.",
     "Based on the text, does it hold that {name} {x}?", "What is true about {name}?",
     ("not under {k}", "past {k}", "not over {k}", "of under {k}", "of just {k}")),
]

# spec: rule 1 of section 4.4 is read per slot across families; joins are also different ("own connectives").
# One dict per family. One condition frame serves bool and threshold conditions, and one conclusion frame and one
# binary question frame serve both polarities (the pool phrase carries the polarity).
_SLOTS = ("label", "rule", "join", "cond", "concl", "exc", "pri", "q", "q_mc")
FAMILIES = {f"t{n:02d}": {**dict(zip(_SLOTS, r)), "ops": dict(zip(OPS, r[9]))} for n, r in enumerate(_ROWS)}


def group(template_id):
    return "tg%d" % (int(template_id[1:]) // 3)


def _roman(n):
    out = ""
    for v, s in ((10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while n >= v:
            out, n = out + s, n - v
    return out


def label(template_id, i):
    """The label of the rule at position i (0-based)."""
    return FAMILIES[template_id]["label"].format(n=i + 1, A=chr(65 + i), R=_roman(i + 1))
