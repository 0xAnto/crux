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

# to_records reads the train and dev splits only.
# The protocol forbids test reads outside Steps 4 and 8. Only the Step 4 split freezer calls to_test_records.
ALLOWED_SPLITS = ("train", "dev")
FILES = [f"meta-{s}.jsonl" for s in ALLOWED_SPLITS]  # the plain train/dev files do not exist in the release
# Option order of each variant (the canonical order of all sources). target is the position of the label.
OPTIONS = {"OWA": ["True", "False", "Unknown"], "CWA": ["True", "False"]}
# The test-only subsets of the OWA release (Step 4): subset name -> folder. ParaRules is the NatLang folder.
TEST_SUBSETS = {"depth5": "depth-5", "birds_electricity": "birds-electricity", "pararules": "NatLang"}


def fetch_zip(cache_dir, url=URL, sha256=SHA256):
    """Download the zip once, check its SHA-256, and return its path."""
    cache_dir = Path(cache_dir)
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
    return zip_path


def extract(zip_path, out_dir, variant, folder, names):
    """Write the files `names` of OWA-like folder `folder` from the zip to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        members = z.namelist()
        for name in names:
            suffix = f"/{variant}/{folder}/{name}"
            (member,) = [n for n in members if n.endswith(suffix) and not n.startswith("__MACOSX")]
            (out_dir / name).write_bytes(z.read(member))


def download(cache_dir, variant="OWA", depth=3, url=URL, sha256=SHA256):
    """Download the zip once. Extract only the train and dev meta files of one variant and depth.

    Return the folder that holds the files.
    """
    out_dir = Path(cache_dir) / variant / f"depth-{depth}"
    if all((out_dir / name).exists() for name in FILES):
        return out_dir
    extract(fetch_zip(cache_dir, url, sha256), out_dir, variant, f"depth-{depth}", FILES)
    return out_dir


def download_test(cache_dir, subset, url=URL, sha256=SHA256):
    """Only the Step 4 split freezer calls this. Extract meta-test.jsonl of one OWA test-only subset.

    Step 8 reads the frozen file `data/splits/test.jsonl`, not this one. Return the folder that holds the file.
    """
    out_dir = Path(cache_dir) / "test-only" / TEST_SUBSETS[subset]
    if not (out_dir / "meta-test.jsonl").exists():
        extract(fetch_zip(cache_dir, url, sha256), out_dir, "OWA", TEST_SUBSETS[subset], ["meta-test.jsonl"])
    return out_dir


def split_theory(row):
    """Return (facts, rules) of one theory in the order of the original theory text.

    The theory text lists all triples (facts) first, then all rules, in the key order.
    The code checks this order against the `theory` string. It raises an error if they differ.

    A ParaRules row (NatLang folder) has a paraphrased `theory`. The `triples` and `rules` keep the plain
    sentences, so the code takes the fact and rule text from `sentences`, and `mappings` links each
    triple or rule to its sentence. One sentence can state several facts. The check needs the facts first
    and the rules after them. It compares the sentences with `theory`, after it turns each ";" into "."
    and merges the white space, because the release writes some sentence breaks as ";" in `sentences`.
    """
    if "sentences" in row:
        facts_of = {row["mappings"][k] for k in row["triples"]}
        rules_of = {row["mappings"][k] for k in row["rules"]}
        keys = sorted(row["sentences"], key=lambda k: int(k[4:]))
        facts, rules = [[row["sentences"][k] for k in keys if k in of] for of in (facts_of, rules_of)]
        norm = lambda text: " ".join(text.replace(";", ".").split())
        if norm(" ".join(facts + rules)) != norm(row["theory"]):
            raise ValueError(f"theory {row['id']}: the sentences do not match the theory text")
        return facts, rules
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


def _records(path, options, dataset_depth, prefix="", **extra_meta):
    """Yield one schema record per question of a meta file. `dataset_depth` is an int or a function of the row."""
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            facts, rules = split_theory(row)
            for key, q in row["questions"].items():
                label = label_name(q["answer"])
                yield {
                    "id": f"{prefix}{row['id']}-{key}", "source": "proofwriter", "task_type": "binary",
                    "state": " ".join(facts),
                    "instruction": " ".join(rules + ["Question: " + q["question"]]),
                    "options": options, "target": options.index(label),
                    "meta": {"depth": int(q["QDep"]), "dataset_depth": dataset_depth(row) if callable(dataset_depth) else dataset_depth,
                             "label": label.lower(), "theory_id": row["id"], "strategy": q["strategy"], **extra_meta},
                }


def to_records(data_dir, split, variant="OWA", depth=3):
    """Step 2 converter: one schema record per ProofWriter question. See docs/schema.md.

    state = the facts. instruction = the rules, then "Question: " and the statement.
    options = OPTIONS[variant]. The default is the open-world set with 3 classes.
    `depth` is the depth of the dataset folder. meta.dataset_depth records it.
    """
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"Split {split!r} is not allowed. Use one of {ALLOWED_SPLITS}.")
    return list(_records(Path(data_dir) / f"meta-{split}.jsonl", OPTIONS[variant], depth, variant=variant, split=split))


def to_test_records(data_dir, subset):
    """Step 4 converter for the test-only subsets (open world, 3 classes). Only the split freezer calls it.

    It writes nothing. The freezer puts the records in the frozen test split, and Step 8 reads that file.
    `subset` is a key of TEST_SUBSETS. The ids start with "pw-<subset>-", so they are unique in one file.
    meta.subset records the subset. meta.dataset_depth is 5 for depth5. The other two folders have no depth number,
    so it is the `maxD` of the theory. For pararules, `state` and `instruction` hold the paraphrased sentences.
    """
    depth = 5 if subset == "depth5" else (lambda row: row["maxD"])
    return list(_records(Path(data_dir) / "meta-test.jsonl", OPTIONS["OWA"], depth, f"pw-{subset}-",
                         variant="OWA", split="test", subset=subset))


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
