"""Download and load RuleTaker (Clark et al. 2020) from the official Ai2 release."""
import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

# The loader reads the train and dev splits only.
# The protocol forbids test reads outside Step 8. The code never extracts a test file.
ALLOWED_SPLITS = ("train", "dev")
# meta-<split>.jsonl holds the facts ("triples") and the rules as separate lists. Step 2 needs them.
FILES = [f"{s}.jsonl" for s in ALLOWED_SPLITS] + [f"meta-{s}.jsonl" for s in ALLOWED_SPLITS]
# Step 2 option order. All sources use the canonical order True, False (see docs/schema.md).
OPTIONS = ["True", "False"]


def download(url, cache_dir, variant, sha256=None):
    """Download the zip once. Extract only the train and dev files (and their meta files) of one variant.

    Return the folder that holds the two files.
    """
    cache_dir = Path(cache_dir)
    out_dir = cache_dir / variant
    if all((out_dir / name).exists() for name in FILES):
        return out_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / Path(url).name
    if not zip_path.exists():
        tmp = zip_path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(zip_path)
    if sha256:
        digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
        if digest != sha256:
            raise ValueError(f"SHA-256 mismatch for {zip_path}: {digest}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        for name in FILES:
            suffix = f"/original/{variant}/{name}"
            (member,) = [n for n in names if n.endswith(suffix) and not n.startswith("__MACOSX")]
            (out_dir / name).write_bytes(z.read(member))
    return out_dir


def parse_row(row, labels):
    """Turn one theory (one JSONL row) into (context, question, label) examples.

    `labels` is the ordered list of label names, for example ["false", "true"].
    The label id is the position of the name in this list.
    """
    ids = {name: i for i, name in enumerate(labels)}
    return [
        {
            "id": q["id"],
            "context": row["context"],  # facts and rules in one string
            "question": q["text"],
            "label": ids[str(q["label"]).lower()],  # true/false (bool) or "Unknown"
        }
        for q in row["questions"]
    ]


def load_split(data_dir, split, labels, max_examples=None):
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"Split {split!r} is not allowed. Use one of {ALLOWED_SPLITS}.")
    examples = []
    with open(Path(data_dir) / f"{split}.jsonl") as f:
        for line in f:
            examples += parse_row(json.loads(line), labels)
            if max_examples and len(examples) >= max_examples:
                return examples[:max_examples]
    return examples


def split_theory(row, meta_row):
    """Return (facts, rules) of one theory in the order of the context sentences.

    The context is a permutation of meta triple1..N then rule1..M. meta.sentenceScramble gives,
    for each context sentence, its 1-based position in that list. A position up to N is a fact.
    """
    facts = [t["text"] for t in meta_row["triples"].values()]
    rules = [r["text"] for r in meta_row["rules"].values()]
    both = facts + rules
    order = row["meta"]["sentenceScramble"]
    if row["id"] != meta_row["id"] or sorted(order) != list(range(1, len(both) + 1)):
        raise ValueError(f"theory {row['id']}: the meta row does not match the context")
    return [both[i - 1] for i in order if i <= len(facts)], [both[i - 1] for i in order if i > len(facts)]


def to_records(data_dir, split, depth=3):
    """Step 2 converter: one schema record per RuleTaker question. See docs/schema.md.

    state = the facts. instruction = the rules, then "Question: " and the statement.
    options = OPTIONS (the canonical order of all sources). Step 1 keeps its own label ids.
    `depth` is the depth of the dataset folder. meta.dataset_depth records it.
    """
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"Split {split!r} is not allowed. Use one of {ALLOWED_SPLITS}.")
    data_dir = Path(data_dir)
    records = []
    with open(data_dir / f"{split}.jsonl") as f, open(data_dir / f"meta-{split}.jsonl") as fm:
        for line, meta_line in zip(f, fm, strict=True):
            row, meta_row = json.loads(line), json.loads(meta_line)
            facts, rules = split_theory(row, meta_row)
            for q in row["questions"]:
                label = str(q["label"]).capitalize()  # bool True/False; an unknown label raises below
                records.append({
                    "id": q["id"], "source": "ruletaker", "task_type": "binary",
                    "state": " ".join(facts),
                    "instruction": " ".join(rules + ["Question: " + q["text"]]),
                    "options": list(OPTIONS), "target": OPTIONS.index(label),
                    "meta": {"depth": q["meta"]["QDep"], "dataset_depth": depth, "label": label.lower(),
                             "theory_id": row["id"], "strategy": q["meta"]["strategy"], "split": split},
                })
    return records


def main():
    """Convert train and dev to data/processed/ruletaker/depth<depth>/<split>.jsonl. The test split is never read.

    The default is depth 3 (questions of depth 0 to 3). Step 4 keeps depth 5 for the test only.
    """
    import yaml

    from crux.schema import PROCESSED_DIR, ROOT, summarize, write_jsonl

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "step1_ruletaker.yaml"))  # url, sha256, cache_dir
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--splits", nargs="+", default=list(ALLOWED_SPLITS))
    args = ap.parse_args()
    d = yaml.safe_load(Path(args.config).read_text())["data"]
    data_dir = download(d["url"], ROOT / d["cache_dir"], f"depth-{args.depth}", d.get("sha256"))
    for split in args.splits:
        records = to_records(data_dir, split, args.depth)
        path = write_jsonl(records, PROCESSED_DIR / "ruletaker" / f"depth{args.depth}" / f"{split}.jsonl")
        print(path, json.dumps(summarize(records)))


if __name__ == "__main__":
    main()
