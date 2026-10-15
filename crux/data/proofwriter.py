"""Download and convert ProofWriter (Tafjord et al. 2021) from the official Ai2 release."""
import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

# Official Ai2 release. DATA_LICENSES.md lists the Ai2 data page.
URL = "https://aristo-data-public.s3.amazonaws.com/proofwriter/proofwriter-dataset-V2020.12.3.zip"
SHA256 = "bbc5694901e8306d0bd659aa1ad53ccfd02c201864f4b320ffa3777827d1fc26"  # of the zip, computed by the author on 2026-10-15

# The converter reads the train and dev splits only.
# The protocol forbids test reads outside Step 8. The code never extracts a test file.
ALLOWED_SPLITS = ("train", "dev")
FILES = [f"meta-{s}.jsonl" for s in ALLOWED_SPLITS]  # the plain train/dev files do not exist in the release
# Option order of each variant (the canonical order of all sources). target is the position of the label.
OPTIONS = {"OWA": ["True", "False", "Unknown"], "CWA": ["True", "False"]}


def download(cache_dir, variant="OWA", depth=3, url=URL, sha256=SHA256):
    """Download the zip once. Extract only the train and dev meta files of one variant and depth.

    Return the folder that holds the files.
    """
    cache_dir = Path(cache_dir)
    out_dir = cache_dir / variant / f"depth-{depth}"
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
            suffix = f"/{variant}/depth-{depth}/{name}"
            (member,) = [n for n in names if n.endswith(suffix) and not n.startswith("__MACOSX")]
            (out_dir / name).write_bytes(z.read(member))
    return out_dir


def split_theory(row):
    """Return (facts, rules) of one theory in the order of the original theory text.

    The theory text lists all triples (facts) first, then all rules, in the key order.
    The code checks this order against the `theory` string. It raises an error if they differ.
    """
    facts = [t["text"] for t in row["triples"].values()]
    rules = [r["text"] for r in row["rules"].values()]
    if " ".join(facts + rules) != row["theory"]:
        raise ValueError(f"theory {row['id']}: the triples and rules do not match the theory text")
    return facts, rules


def label_name(answer):
    """Return the label name 'True', 'False', or 'Unknown'. The source holds bool or the string 'Unknown'."""
    if isinstance(answer, bool):
        return "True" if answer else "False"
    if answer == "Unknown":
        return "Unknown"
    raise ValueError(f"unexpected answer {answer!r}")


def to_records(data_dir, split, variant="OWA", depth=3):
    """Step 2 converter: one schema record per ProofWriter question. See docs/schema.md.

    state = the facts. instruction = the rules, then "Question: " and the statement.
    options = OPTIONS[variant]. The default is the open-world set with 3 classes.
    `depth` is the depth of the dataset folder. meta.dataset_depth records it.
    """
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"Split {split!r} is not allowed. Use one of {ALLOWED_SPLITS}.")
    options = OPTIONS[variant]
    records = []
    with open(Path(data_dir) / f"meta-{split}.jsonl") as f:
        for line in f:
            row = json.loads(line)
            facts, rules = split_theory(row)
            for key, q in row["questions"].items():
                label = label_name(q["answer"])
                records.append({
                    "id": f"{row['id']}-{key}", "source": "proofwriter", "task_type": "binary",
                    "state": " ".join(facts),
                    "instruction": " ".join(rules + ["Question: " + q["question"]]),
                    "options": options, "target": options.index(label),
                    "meta": {"depth": int(q["QDep"]), "dataset_depth": depth, "label": label.lower(), "theory_id": row["id"],
                             "strategy": q["strategy"], "variant": variant, "split": split},
                })
    return records


def main():
    """Convert train and dev to data/processed/proofwriter/<variant>-depth<depth>/<split>.jsonl.

    The test split is never read. The default is depth 3 (questions of depth 0 to 3).
    Step 4 keeps depth 5 for the test only.
    """
    from crux.schema import PROCESSED_DIR, ROOT, summarize, write_jsonl

    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(ROOT / "data" / "proofwriter"))  # not in git
    ap.add_argument("--variant", default="OWA", choices=list(OPTIONS))
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--splits", nargs="+", default=list(ALLOWED_SPLITS))
    args = ap.parse_args()
    data_dir = download(args.cache_dir, args.variant, args.depth)
    for split in args.splits:
        records = to_records(data_dir, split, args.variant, args.depth)
        path = write_jsonl(records, PROCESSED_DIR / "proofwriter" / f"{args.variant}-depth{args.depth}" / f"{split}.jsonl")
        print(path, json.dumps(summarize(records)))


if __name__ == "__main__":
    main()
