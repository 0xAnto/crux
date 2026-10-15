"""Step 2: the unified decision schema, its validation, and JSONL storage.

Each record is one decision. See docs/schema.md for the fields and the converter contract.
Validate one or more files: python -m crux.schema data/processed/ruletaker/depth3/train.jsonl
"""
import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import TypedDict

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"  # not in git
TASK_TYPES = ("binary", "multi_choice", "scoring")
# Optional meta keys that later steps read, with their types. Other meta keys are free.
META_TYPES = {"template_id": str, "composition_id": str, "vocabulary_id": str, "depth": int, "dataset_depth": int,
              "label": str}
SOURCE_RE = re.compile(r"[a-z0-9_]+")  # the source is also a folder name


class Record(TypedDict):
    id: str            # unique in its file
    source: str        # dataset name, for example "ruletaker"
    task_type: str     # one of TASK_TYPES
    state: str         # facts and evidence
    instruction: str   # rules and question
    options: list      # candidate answers (str), 2 or more
    target: object     # int option index (binary, multi_choice) or list of float, one per option (scoring)
    meta: dict         # optional ids for the splits (META_TYPES) and source details


def validate(rec):
    """Raise ValueError with a clear message if the record breaks the schema. Return the record."""
    def fail(msg):
        raise ValueError(f"record {rec.get('id', '?') if isinstance(rec, dict) else '?'}: {msg}")

    if not isinstance(rec, dict):
        fail(f"a record must be a dict, not {type(rec).__name__}")
    missing = set(Record.__annotations__) - set(rec)
    extra = set(rec) - set(Record.__annotations__)
    if missing or extra:
        fail(f"missing fields {sorted(missing)}, unknown fields {sorted(extra)}")
    for key in ("id", "source", "task_type", "state", "instruction"):
        if not isinstance(rec[key], str) or not rec[key].strip():
            fail(f"'{key}' must be a non-empty string")
    if not SOURCE_RE.fullmatch(rec["source"]):
        fail(f"'source' must match {SOURCE_RE.pattern}, not {rec['source']!r}")
    task, options, target = rec["task_type"], rec["options"], rec["target"]
    if task not in TASK_TYPES:
        fail(f"'task_type' must be one of {TASK_TYPES}, not {task!r}")

    if not isinstance(options, list) or len(options) < 2:
        fail("'options' must be a list with 2 or more items")
    if not all(isinstance(o, str) and o.strip() for o in options):
        fail("each option must be a non-empty string")
    keys = [o.strip().casefold() for o in options]
    if len(set(keys)) != len(keys):
        fail(f"duplicate options {sorted(k for k, n in Counter(keys).items() if n > 1)}")
    if task == "binary" and len(options) > 3:
        fail(f"a binary task has 2 options, or 3 with an 'Unknown' class, not {len(options)}")
    if task == "binary" and len(options) == 3 and keys[2] != "unknown":
        fail(f"the third option of a binary task must be 'Unknown', not {options[2]!r}")

    if task == "scoring":
        if not isinstance(target, list) or len(target) != len(options):
            fail("a scoring target must be a list with one score per option")
        if not all(isinstance(s, (int, float)) and not isinstance(s, bool) and math.isfinite(s) for s in target):
            fail("each score must be a finite number")
    else:
        if not isinstance(target, int) or isinstance(target, bool):
            fail(f"a {task} target must be an int option index, not {type(target).__name__}")
        if not 0 <= target < len(options):
            fail(f"target {target} is out of range for {len(options)} options")

    if not isinstance(rec["meta"], dict):
        fail("'meta' must be a dict")
    for key, typ in META_TYPES.items():
        value = rec["meta"].get(key)
        if value is not None and (not isinstance(value, typ) or isinstance(value, bool)):
            fail(f"meta '{key}' must be {typ.__name__}, not {type(value).__name__}")
    label = rec["meta"].get("label")
    if label is not None and task != "scoring" and keys[target] != label.strip().casefold():
        fail(f"options[target] is {options[target]!r}, but meta 'label' is {label!r}")
    return rec


def write_jsonl(records, path):
    """Validate all records, check that the ids are unique, then write one JSON record per line."""
    records = [validate(r) for r in records]
    dupes = [i for i, n in Counter(r["id"] for r in records).items() if n > 1]
    if dupes:
        raise ValueError(f"{len(dupes)} duplicate ids, for example {dupes[:3]}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with open(tmp, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.rename(path)
    return path


def read_jsonl(path):
    """Read the records of a JSONL file. Validate each record and the unique ids."""
    records = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = validate(json.loads(line))
            except ValueError as e:  # json.JSONDecodeError is a ValueError
                raise ValueError(f"{path}:{n}: {e}") from None
            records.append(rec)
    dupes = [i for i, n in Counter(r["id"] for r in records).items() if n > 1]
    if dupes:
        raise ValueError(f"{path}: {len(dupes)} duplicate ids, for example {dupes[:3]}")
    return records


def summarize(records):
    """Count the records per source, task type, number of options, and label."""
    def label(r):  # binary: the option text; multi_choice: the target position; scoring: no label
        if r["task_type"] == "binary":
            return r["options"][r["target"]]
        return f"index {r['target']}" if r["task_type"] == "multi_choice" else "(scores)"
    return {
        "n": len(records),
        "source": dict(Counter(r["source"] for r in records)),
        "task_type": dict(Counter(r["task_type"] for r in records)),
        "num_options": dict(sorted(Counter(len(r["options"]) for r in records).items())),
        "label": dict(Counter(label(r) for r in records).most_common()),
    }


def main():
    ap = argparse.ArgumentParser(description="Validate JSONL files of the decision schema and print counts.")
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    for path in args.files:
        try:
            records = read_jsonl(path)
        except ValueError as e:
            raise SystemExit(f"INVALID {e}")
        print(path, json.dumps(summarize(records), indent=1))


if __name__ == "__main__":
    main()
