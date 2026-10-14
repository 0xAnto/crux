"""Download and load RuleTaker (Clark et al. 2020) from the official Ai2 release."""
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

# The loader reads the train and dev splits only.
# The protocol forbids test reads outside Step 8. The code never extracts a test file.
ALLOWED_SPLITS = ("train", "dev")


def download(url, cache_dir, variant, sha256=None):
    """Download the zip once. Extract only train.jsonl and dev.jsonl of one variant.

    Return the folder that holds the two files.
    """
    cache_dir = Path(cache_dir)
    out_dir = cache_dir / variant
    if all((out_dir / f"{s}.jsonl").exists() for s in ALLOWED_SPLITS):
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
        for split in ALLOWED_SPLITS:
            suffix = f"/original/{variant}/{split}.jsonl"
            (member,) = [n for n in names if n.endswith(suffix)]
            (out_dir / f"{split}.jsonl").write_bytes(z.read(member))
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
