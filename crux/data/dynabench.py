"""Convert DynaBench (DynaGuard paper, 2025) to schema records. Evaluation only.

Source: Hugging Face dataset `montehoover/DynaBench` (MIT), subset `DynaBench`, split `test` (543 examples).
This is not the Facebook Dynabench platform. The code does not use the SafetyMix subset
(its upstream licenses are not checked). See DATA_LICENSES.md.
"""
import argparse
import hashlib
import json
from pathlib import Path

# The pinned commit of the dataset repository. It makes the download reproducible.
REPO_ID = "montehoover/DynaBench"
REVISION = "b5bf2061252ca634708c6bbd178c1609505ba6cd"
FILE = "DynaBench/test.parquet"
# DynaBench is evaluation only in CRUX. The only split is the test split.
ALLOWED_SPLITS = ("test",)
# The order is fixed for all records. The target is the index in this list.
OPTIONS = ["compliant", "not compliant"]
LABEL_IDS = {"PASS": 0, "FAIL": 1}  # PASS = the dialogue follows the policy
QUESTION = "Question: Does the dialogue comply with the policy?"


def download(cache_dir):
    """Download the test file at the pinned revision. Return the folder that holds `DynaBench/test.parquet`."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(REPO_ID, FILE, repo_type="dataset", revision=REVISION, local_dir=cache_dir)
    return Path(cache_dir)


def to_records(data_dir, split="test"):
    """Step 2 converter: one schema record per DynaBench example. See docs/schema.md.

    state = the dialogue, one turn per line ("User: ..." / "Agent: ..."), as in the source.
    instruction = the policy (the rules), then the fixed question.
    options = ["compliant", "not compliant"]. The source label PASS gives target 0, FAIL gives target 1.
    """
    import pandas as pd

    if split not in ALLOWED_SPLITS:
        raise ValueError(f"Split {split!r} is not allowed. Use one of {ALLOWED_SPLITS}.")
    df = pd.read_parquet(Path(data_dir) / FILE)
    records = []
    for row in df.itertuples(index=False):
        info = json.loads(row.metadata)
        records.append({
            "id": row.base_id, "source": "dynabench", "task_type": "binary",
            "state": row.transcript.strip(),
            "instruction": f"{row.policy.strip()}\n\n{QUESTION}",
            "options": list(OPTIONS), "target": LABEL_IDS[row.label],
            # meta.label must equal options[target] (crux/schema.py). source_label keeps PASS or FAIL.
            "meta": {"label": OPTIONS[LABEL_IDS[row.label]], "source_label": row.label, "policy_id": hashlib.sha256(row.policy.encode()).hexdigest()[:12],
                     "num_rules": info["num_rules"], "num_turns": info["num_turns"], "domain": info["domain"],
                     "revision": REVISION, "split": split, "eval_only": True},
        })
    return records


def main():
    """Download the test split and convert it to data/processed/dynabench/test.jsonl."""
    from crux.schema import PROCESSED_DIR, ROOT, summarize, write_jsonl

    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(ROOT / "data" / "raw" / "dynabench"))
    args = ap.parse_args()
    records = to_records(download(args.cache_dir), "test")
    path = write_jsonl(records, PROCESSED_DIR / "dynabench" / "test.jsonl")
    print(path, json.dumps(summarize(records)))


if __name__ == "__main__":
    main()
