"""Download and convert BoardgameQA (Kazemi et al. 2023) from the official Google release.

BoardgameQA is EVALUATION-ONLY in CRUX. No model trains on it, and no model sees it before Step 8.

Task: binary, with 3 fixed options in this order: ["proved", "disproved", "unknown"].
The Binary-* subsets have only proved/disproved labels. Their records have 2 options: ["proved", "disproved"].
`target` is the index of the original label in the options of the record.

Raw record fields: facts, rules, preferences (rule priorities, can be empty), example (the full prompt),
goal, proof, theory, label. The release has no record ids.
  state       = the facts (the game state).
  instruction = the rules, the preferences, and "Question: <question>".
  The question is the text of `example` after QUESTION_MARKER. The converter checks that the marker
  occurs once and that the question ends with "?".

Folders: one folder per subset, named BoardgameQA-<variant>-depth<N>. Files: train, valid, test (.json).
The output path is data/processed/boardgameqa/<subset>/<split>.jsonl, for example Main-depth2/valid.jsonl.
"""
import argparse
import hashlib
import json
import re
import urllib.request
import zipfile
from pathlib import Path

URL = "https://storage.googleapis.com/gresearch/BoardgameQA/BoardgameQA.zip"
SHA256 = "4d8aa980b3d67be449a3a32f75b0cfa682c477f5924b2bf55740c8d93a91474b"
OPTIONS = ["proved", "disproved", "unknown"]
SPLITS = ("train", "valid", "test")  # the release names the dev split "valid"
QUESTION_MARKER = "Based on the game state and the rules and preferences, "
SUBSET_RE = re.compile(r"BoardgameQA-(?P<variant>[A-Za-z]+)-depth(?P<depth>\d+)")
CACHE_DIR = Path("data") / "boardgameqa"


def download(cache_dir, url=URL, sha256=SHA256):
    """Download the zip once, check the SHA-256, and extract the JSON files. Return the folder of the subsets."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / Path(url).name
    if not zip_path.exists():
        tmp = zip_path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(zip_path)
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    if digest != sha256:
        raise ValueError(f"SHA-256 mismatch for {zip_path}: {digest}")
    out_dir = cache_dir / "BoardgameQA"
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            parts = name.split("/")
            # Write only BoardgameQA/<subset>/<split>.json. This also blocks unsafe member paths.
            if len(parts) == 3 and parts[0] == "BoardgameQA" and SUBSET_RE.fullmatch(parts[1]) \
                    and parts[2] in {f"{s}.json" for s in SPLITS}:
                target = out_dir / parts[1] / parts[2]
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(z.read(name))
    return out_dir


def list_subsets(data_dir):
    """Return the subset folder names (for example "BoardgameQA-Main-depth2") in sorted order."""
    return sorted(p.name for p in Path(data_dir).iterdir() if p.is_dir() and SUBSET_RE.fullmatch(p.name))


def split_example(row):
    """Return (state, instruction, question) of one raw row."""
    head, marker, question = row["example"].partition(QUESTION_MARKER)
    if not marker or QUESTION_MARKER in question or not question.endswith("?"):
        raise ValueError("the example text does not end with one question")
    parts = [row["rules"].strip(), row["preferences"].strip(), "Question: " + question.strip()]
    return row["facts"].strip(), " ".join(p for p in parts if p), question.strip()


def to_records(data_dir, split, subset):
    """Step 2 converter: one schema record per BoardgameQA example. See docs/schema.md.

    `data_dir` holds the subset folders. `subset` is a folder name, for example "BoardgameQA-Main-depth2".
    The id is "<short subset>-<split>-<row number>", because the release has no ids.
    """
    if split not in SPLITS:
        raise ValueError(f"Split {split!r} is not allowed. Use one of {SPLITS}.")
    m = SUBSET_RE.fullmatch(subset)
    if not m:
        raise ValueError(f"Unknown subset name {subset!r}")
    short = f"{m['variant']}-depth{m['depth']}"
    rows = json.loads((Path(data_dir) / subset / f"{split}.json").read_text())
    options = OPTIONS[:2] if m["variant"] == "Binary" else OPTIONS
    records = []
    for i, row in enumerate(rows):
        state, instruction, _ = split_example(row)
        records.append({
            "id": f"{short}-{split}-{i:05d}", "source": "boardgameqa", "task_type": "binary",
            "state": state, "instruction": instruction,
            "options": list(options), "target": options.index(row["label"]),
            "meta": {"depth": int(m["depth"]), "label": row["label"], "subset": short, "variant": m["variant"],
                     "has_preferences": bool(row["preferences"].strip()), "split": split,
                     "evaluation_only": True},
        })
    return records


def main():
    """Convert all subsets and splits to data/processed/boardgameqa/<subset>/<split>.jsonl."""
    from crux.schema import PROCESSED_DIR, ROOT, summarize, write_jsonl

    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(ROOT / CACHE_DIR))
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    args = ap.parse_args()
    data_dir = download(args.cache_dir)
    for subset in list_subsets(data_dir):
        short = subset.removeprefix("BoardgameQA-")
        for split in args.splits:
            records = to_records(data_dir, split, subset)
            path = write_jsonl(records, PROCESSED_DIR / "boardgameqa" / short / f"{split}.jsonl")
            print(path, json.dumps(summarize(records)))


if __name__ == "__main__":
    main()
