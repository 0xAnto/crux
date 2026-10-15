"""Convert the Super-NaturalInstructions held-out classification tasks (Wang et al. 2022) to schema records.

EVALUATION ONLY. CRUX does not train on this data. No model sees it before Step 8.
Do not redistribute the instances: data/ is git-ignored. Publish task ids and hashes only.

Source: the official repository https://github.com/allenai/natural-instructions at the pinned commit
below. The converter reads only the tasks in splits/default/test_tasks.txt (119 held-out tasks).

Selection rule. A task is kept only if all of these are true (`select_task` applies them in code):
1. License: the task field "Instance License" does not contain "Unknown".
2. Category: a name in the task field "Categories" ends with "Classification", or is
   "Textual Entailment" or "Dialogue Act Recognition" (CLASSIFICATION_CATEGORIES).
3. Output form: every instance has exactly one output string.
4. Label set: the outputs form a fixed set of 2 to 8 labels (MIN_LABELS, MAX_LABELS).
   The labels are different after case folding and outer-space removal.
   Each label has at least MIN_PER_LABEL instances on average (the number of instances divided by the
   number of labels). This rule removes the tasks with free-text outputs.
The options are the labels in sorted order. The target is the index of the instance output.

Run: python -m crux.data.supernat   (needs git and network once, for the pinned commit)
"""
import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

REPO_URL = "https://github.com/allenai/natural-instructions"
COMMIT = "55a365637381ce7f3748fa2eac7aef1a113bbb82"  # master on 2026-10-15
SPLIT_FILE = "splits/default/test_tasks.txt"
ALLOWED_SPLITS = ("test",)  # the held-out tasks are the only split. Use it for evaluation only.

CLASSIFICATION_CATEGORIES = ("Textual Entailment", "Dialogue Act Recognition")
MIN_LABELS, MAX_LABELS = 2, 8
MIN_PER_LABEL = 20
QUESTION = "Question: Which option is the correct output for the input?"


def download(repo_dir, commit=COMMIT):
    """Fetch only the pinned commit (depth 1) into repo_dir. Return repo_dir."""
    repo_dir = Path(repo_dir)
    if (repo_dir / SPLIT_FILE).exists():
        return repo_dir
    repo_dir.mkdir(parents=True, exist_ok=True)
    for cmd in (["init", "-q"], ["remote", "add", "origin", REPO_URL],
                ["fetch", "-q", "--depth", "1", "origin", commit], ["checkout", "-q", "FETCH_HEAD"]):
        subprocess.run(["git", "-C", str(repo_dir), *cmd], check=True)
    return repo_dir


def is_classification_category(categories):
    return any(c.endswith("Classification") or c in CLASSIFICATION_CATEGORIES for c in categories)


def label_set(task):
    """Return the sorted labels, or None if the outputs do not form a fixed small label set."""
    instances = task["Instances"]
    if any(not isinstance(i["output"], list) or len(i["output"]) != 1 for i in instances):
        return None
    counts = Counter(i["output"][0] for i in instances)
    labels = sorted(counts)
    if len({s.strip().casefold() for s in labels}) != len(labels) or any(not s.strip() for s in labels):
        return None
    if not MIN_LABELS <= len(labels) <= MAX_LABELS or len(instances) / len(labels) < MIN_PER_LABEL:
        return None
    return labels


def select_task(task):
    """Return (labels, reason). labels is None if the task is rejected; reason names the first failed rule."""
    if "Unknown" in task["Instance License"]:
        return None, "license_unknown"
    if not is_classification_category(task["Categories"]):
        return None, "category"
    labels = label_set(task)
    if labels is None:
        return None, "label_set"
    return labels, "ok"


def is_non_commercial(task):
    return any("-NC" in lic for lic in task["Instance License"])


def load_tasks(data_dir):
    """Yield (task_id, task) for the held-out tasks, in the order of the split file."""
    data_dir = Path(data_dir)
    for line in (data_dir / SPLIT_FILE).read_text().splitlines():
        if line.strip():
            yield line.strip(), json.loads((data_dir / "tasks" / f"{line.strip()}.json").read_text())


def convert_task(task_id, task, labels):
    """Return the schema records of one selected task."""
    index = {s: i for i, s in enumerate(labels)}
    definition = " ".join(task["Definition"]).strip()
    license_ = "; ".join(task["Instance License"])
    return [{
        "id": inst["id"], "source": "supernat", "task_type": "multi_choice",
        "state": inst["input"],
        "instruction": f"{definition} {QUESTION}",
        "options": list(labels), "target": index[inst["output"][0]],
        "meta": {"task_id": task_id, "category": "; ".join(task["Categories"]), "license": license_,
                 "label": inst["output"][0], "eval_only": True},
    } for inst in task["Instances"]]


def to_records(data_dir, split, loaded=None):
    """Step 2 converter: one schema record per instance of each selected task. See docs/schema.md.

    state = the instance input. instruction = the task definition, then "Question: ...".
    options = the label set of the task. The few-shot examples of the task are not used.
    `loaded` is an optional list of (task_id, task) to skip the second read of the files.
    """
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"Split {split!r} is not allowed. Use one of {ALLOWED_SPLITS}.")
    records = []
    for task_id, task in loaded if loaded is not None else load_tasks(data_dir):
        labels, _ = select_task(task)
        if labels is not None:
            records += convert_task(task_id, task, labels)
    return records


def main():
    """Convert the held-out classification tasks to data/processed/supernat/test.jsonl and tasks.json."""
    from crux.schema import PROCESSED_DIR, ROOT, summarize, write_jsonl

    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", default=str(ROOT / "data" / "raw" / "supernat" / ".repo"))  # the dot hides it from pytest
    args = ap.parse_args()
    data_dir = download(args.repo_dir)
    loaded = list(load_tasks(data_dir))
    reasons = Counter(select_task(t)[1] for _, t in loaded)
    records = to_records(data_dir, "test", loaded)
    per_task = Counter(r["meta"]["task_id"] for r in records)
    tasks = []
    for task_id, task in loaded:
        labels, _ = select_task(task)
        if labels is not None:
            tasks.append({"task_id": task_id, "labels": labels, "license": task["Instance License"],
                          "non_commercial": is_non_commercial(task), "n_records": per_task[task_id],
                          "sha256": hashlib.sha256(json.dumps(task["Instances"], sort_keys=True).encode()).hexdigest()})
    out = PROCESSED_DIR / "supernat"
    path = write_jsonl(records, out / "test.jsonl")
    (out / "tasks.json").write_text(json.dumps({"repo": REPO_URL, "commit": COMMIT, "tasks": tasks}, indent=1))
    print(path, json.dumps(summarize(records)))
    no_license_rule = [t for _, t in loaded
                       if is_classification_category(t["Categories"]) and label_set(t) is not None]
    print("classification tasks before the license rule", len(no_license_rule),
          "of them with license Unknown", sum("Unknown" in t["Instance License"] for t in no_license_rule))
    print("held-out tasks", len(loaded), "rejected by first failed rule", dict(reasons),
          "selected", len(tasks), "non-commercial", sum(t["non_commercial"] for t in tasks))
    print("records per task", {t["task_id"]: t["n_records"] for t in tasks})
    print("tasks per option count", dict(Counter(len(t["labels"]) for t in tasks)))


if __name__ == "__main__":
    main()
