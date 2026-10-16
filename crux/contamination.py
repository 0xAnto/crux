"""Measure the n-gram overlap of test records with a text corpus (Step 4 carry-over, docs/contamination.md).

The metric is the same as in the Step 4 leakage check. A text becomes the lowercase tokens `[a-z0-9]+`.
Its n-grams are the windows of n consecutive tokens. A record has the text `state + " " + instruction`.
The share of a record is (distinct n-grams of the record found in the corpus) / (distinct n-grams of the record).
The scan flags a record if the share is more than `max_overlap`. A record with fewer than n tokens has no
n-gram and is never flagged. n and `max_overlap` come from `splits.leakage` in configs/protocol.yaml. If the
values are null, the defaults N and MAX_OVERLAP apply.

Method. The scan indexes the n-grams of the TEST records and streams the corpus through the index.
1. Each test token gets an id. A corpus token that is not in a test record gets id 0.
2. A corpus window with an id 0 cannot match, so the scan drops it at once. The scan checks only the other windows.
3. The scan hashes each remaining window (64 bit), looks it up in the sorted test hashes,
   and compares the ids of each candidate with the test n-gram. The result is exact.
4. Documents are separated by id 0, so no window crosses a document.

Usage (the corpus is streamed, never saved to disk; a finished file is not scanned again):
    python -m crux.contamination scan --name NAME --test split.jsonl [...] [--sni task966 task967] --source dolma
    python -m crux.contamination report --name NAME
`--source` is `dolma` (the 66 files of Dolma v1.7 `tulu_flan`), `ettin` (the 65 chunked archives in the Ettin
data repository), or the paths and URLs of corpus files (`.json.gz` or `.tar.gz` with JSON lines and a `text` field).
The results are in runs/contamination/NAME/ (git-ignored): index/, hits/, state.json, results.json, counts.csv.
"""
import argparse
import csv
import gzip
import hashlib
import json
import re
import tarfile
import time
import urllib.request
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
N, MAX_OVERLAP = 8, 0.10  # defaults, used if configs/protocol.yaml has null values
TOKEN = re.compile(r"[a-z0-9]+|\x00")  # \x00 separates the documents
MULT = np.random.default_rng(0).integers(1, 2**64 - 1, size=32, dtype=np.uint64) | np.uint64(1)
BATCH_CHARS = 4_000_000
DOLMA_LIST = "https://huggingface.co/datasets/allenai/dolma/resolve/main/urls/v1_7.txt"
ETTIN_TREE = "https://huggingface.co/api/datasets/jhu-clsp/ettin-pretraining-data/tree/main/tulu_flan?limit=1000"
ETTIN_FILE = "https://huggingface.co/datasets/jhu-clsp/ettin-pretraining-data/resolve/main/"


def settings(path=ROOT / "configs" / "protocol.yaml"):
    """Return (n, max_overlap) from splits.leakage. Null values give the defaults."""
    leak = yaml.safe_load(Path(path).read_text())["splits"]["leakage"]
    return leak.get("ngram_n") or N, leak.get("max_overlap") or MAX_OVERLAP


def tokens(text):
    return TOKEN.findall(text.lower())


def windows(ids, n):
    """Return the start positions and the hashes of the windows of n ids that have no id 0."""
    m = len(ids) - n + 1
    if m <= 0:
        return np.empty(0, np.int64), np.empty(0, np.uint64)
    zeros = np.concatenate(([0], np.cumsum(ids == 0)))
    start = np.flatnonzero(zeros[n:] == zeros[:m])
    h = np.zeros(len(start), np.uint64)
    for k in range(n):
        h += ids[start + k].astype(np.uint64) * MULT[k]
    return start, h


# ---- records -------------------------------------------------------------------------------------------------

def jsonl_records(path):
    """Yield (id, group, text) for a split file. The group is `meta.subset`, or `source` if the record has no subset."""
    for line in Path(path).open():
        r = json.loads(line)
        yield r["id"], r.get("meta", {}).get("subset") or r["source"], r["state"] + " " + r["instruction"]


def sni_records(tasks_dir, names):
    """Yield (id, group, text) for the instances of the Super-NaturalInstructions tasks, for example task966."""
    for name in names:
        (path,) = Path(tasks_dir).glob(name + "_*.json")
        for inst in json.loads(path.read_text())["Instances"]:
            yield inst["id"], name, inst["input"]


# ---- index ---------------------------------------------------------------------------------------------------

def build_index(records, out, n):
    """Index the distinct n-grams of the records. Write the arrays to the folder `out`. Return the index key."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    vocab, flat, starts, meta = {}, [], [], []
    for rid, group, text in records:
        starts.append(len(flat))
        flat += [vocab.setdefault(t, len(vocab) + 1) for t in tokens(text)] + [0]
        meta.append((rid, group))
    flat = np.array(flat, np.int64)
    pos, h = windows(flat, n)
    H, first, inv = np.unique(h, return_index=True, return_inverse=True)
    G = flat[pos[first][:, None] + np.arange(n)].astype(np.uint32)
    rec = np.searchsorted(starts, pos, "right") - 1
    pair = np.unique(rec * len(H) + inv)  # distinct (record, n-gram) pairs
    np.save(out / "H.npy", H)
    np.save(out / "G.npy", G)
    np.save(out / "rec.npy", pair // len(H))
    np.save(out / "idx.npy", pair % len(H))
    (out / "vocab.json").write_text(json.dumps(vocab))
    (out / "records.json").write_text(json.dumps(meta))
    return hashlib.sha256(H.tobytes() + str(n).encode()).hexdigest()[:16]


_ix = {}


def load_index(out):
    out = Path(out)
    _ix.update(H=np.load(out / "H.npy"), G=np.load(out / "G.npy", mmap_mode="r"), vocab=json.loads((out / "vocab.json").read_text()))
    _ix["n"] = _ix["G"].shape[1]


# ---- corpus --------------------------------------------------------------------------------------------------

def corpus_files(source):
    """Return the file URLs of a corpus: `dolma`, `ettin`, or a list of paths and URLs."""
    if source == ["dolma"]:
        return [u for u in urllib.request.urlopen(DOLMA_LIST).read().decode().split() if "/tulu_flan/" in u]
    if source == ["ettin"]:
        tree = json.load(urllib.request.urlopen(ETTIN_TREE))
        return [ETTIN_FILE + t["path"] for t in tree if t["path"].endswith(".tar.gz")]
    return source


def texts(src):
    """Stream the `text` fields of one corpus file (a local path or a URL)."""
    agent = {"User-Agent": "crux-contamination/1"}  # the server rejects the default agent
    with urllib.request.urlopen(urllib.request.Request(src, headers=agent), timeout=120) if src.startswith("http") else open(src, "rb") as f:
        if src.endswith(".tar.gz"):
            tar = tarfile.open(fileobj=f, mode="r|gz")
            for member in tar:
                if member.isfile() and member.name.endswith(".jsonl") and not member.name.rsplit("/", 1)[-1].startswith("._"):
                    yield from (json.loads(line)["text"] for line in tar.extractfile(member))
        else:
            yield from (json.loads(line)["text"] for line in gzip.open(f))


def scan_batch(batch, hit):
    """Mark in `hit` the test n-grams that occur in the batch of texts. Return the number of tokens."""
    H, G, n = _ix["H"], _ix["G"], _ix["n"]
    joined = "\x00".join(batch)
    if joined.count("\x00") != len(batch) - 1:  # a text holds a \x00 itself
        joined = "\x00".join(t.replace("\x00", " ") for t in batch)
    toks = TOKEN.findall(joined.lower())
    ids = np.fromiter(map(_ix["vocab"].get, toks, repeat(0)), np.int64, len(toks))
    start, h = windows(ids, n)
    at = np.minimum(np.searchsorted(H, h), len(H) - 1)
    cand = H[at] == h
    start, at = start[cand], at[cand]
    same = (G[at] == ids[start[:, None] + np.arange(n)]).all(1)  # exact check: the ids are equal
    hit[at[same]] = True
    return len(toks)


def scan_file(args):
    """Scan one corpus file. Save the indices of the found n-grams. Return the statistics."""
    src, out = args
    for attempt in range(8):
        try:
            t0, hit, docs, ntok, batch, chars = time.time(), np.zeros(len(_ix["H"]), bool), 0, 0, [], 0
            for text in texts(src):
                batch.append(text)
                chars += len(text)
                docs += 1
                if chars >= BATCH_CHARS:
                    ntok += scan_batch(batch, hit)
                    batch, chars = [], 0
            if batch:
                ntok += scan_batch(batch, hit)
            break
        except OSError as e:
            if attempt == 7:
                raise RuntimeError(f"{src}: {e!r}") from None  # a plain error can cross the process border
            time.sleep(30 * (attempt + 1))
    name = src.rsplit("/", 1)[-1]
    np.save(Path(out) / "hits" / (name + ".npy"), np.flatnonzero(hit).astype(np.uint32))
    return {"file": src, "docs": docs, "tokens": ntok, "found": int(hit.sum()), "seconds": round(time.time() - t0, 1)}


# ---- report --------------------------------------------------------------------------------------------------

def report(out, max_overlap):
    """Combine the hits of the finished files. Write counts.csv and results.json. Return the results."""
    out = Path(out)
    state = json.loads((out / "state.json").read_text())
    meta = json.loads((out / "index" / "records.json").read_text())
    rec, idx = np.load(out / "index" / "rec.npy"), np.load(out / "index" / "idx.npy")
    hit = np.zeros(int(idx.max()) + 1 if len(idx) else 0, bool)
    for f in (out / "hits").glob("*.npy"):
        if f.name[:-4] in state["done"]:
            hit[np.load(f)] = True
    total = np.bincount(rec, minlength=len(meta))
    found = np.bincount(rec, weights=hit[idx], minlength=len(meta)).astype(int)
    share = found / np.maximum(total, 1)
    with (out / "counts.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "group", "n_ngrams", "n_found", "share"])
        w.writerows([m[0], m[1], t, c, round(s, 4)] for m, t, c, s in zip(meta, total, found, share))
    flagged = share > max_overlap
    res = {"n": state["n"], "max_overlap": max_overlap, "files_done": len(state["done"]), "files_total": state["files_total"],
           "tokens_scanned": sum(d["tokens"] for d in state["done"].values()), "groups": {}, "flagged_ids": {}}
    for g in sorted({m[1] for m in meta}):
        sel = np.array([m[1] == g for m in meta])
        res["groups"][g] = {"records": int(sel.sum()), "with_ngrams": int((total[sel] > 0).sum()), "any_found": int((found[sel] > 0).sum()),
                            "flagged": int(flagged[sel].sum()), "max_share": round(float(share[sel].max()), 4),
                            "p99_share": round(float(np.percentile(share[sel], 99)), 4)}
        res["flagged_ids"][g] = [m[0] for m, f in zip(meta, flagged) if f and m[1] == g]
    (out / "results.json").write_text(json.dumps(res, indent=1))
    return res


def scan(a):
    n, max_overlap = settings()
    out = ROOT / "runs" / "contamination" / a.name
    (out / "hits").mkdir(parents=True, exist_ok=True)
    records = [r for p in a.test for r in jsonl_records(p)]
    if a.sni:
        records += list(sni_records(a.sni_dir, a.sni))
    key = build_index(records, out / "index", n)
    files = corpus_files(a.source)
    sp = out / "state.json"
    state = json.loads(sp.read_text()) if sp.exists() else {}
    if state.get("key") != key or state.get("files_total") != len(files):
        state = {"key": key, "n": n, "files_total": len(files), "done": {}}  # the test set or the corpus changed
    todo = [f for f in files if f.rsplit("/", 1)[-1] not in state["done"]]
    print(f"{len(records)} records, n={n}, {len(files) - len(todo)}/{len(files)} files done, {len(todo)} to scan", flush=True)
    with Pool(a.workers, load_index, (out / "index",)) as pool:
        for s in pool.imap_unordered(scan_file, [(f, out) for f in todo]):
            state["done"][s["file"].rsplit("/", 1)[-1]] = s
            sp.write_text(json.dumps(state))
            print(f"{s['file'].rsplit('/', 1)[-1]}: {s['docs']} docs, {s['tokens'] / 1e6:.0f}M tokens, {s['found']} n-grams found, "
                  f"{s['tokens'] / s['seconds'] / 1e6:.2f}M tokens/s, {len(state['done'])}/{len(files)}", flush=True)
    print(json.dumps(report(out, max_overlap)["groups"], indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("command", choices=["scan", "report"])
    p.add_argument("--name", required=True)
    p.add_argument("--test", nargs="*", default=[], help="split files (JSON lines)")
    p.add_argument("--sni", nargs="*", default=[], help="Super-NaturalInstructions task names, for example task966")
    p.add_argument("--sni-dir", default=str(ROOT / "data" / "raw" / "supernat" / ".repo" / "tasks"))
    p.add_argument("--source", nargs="+", default=["dolma"])
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    if a.command == "scan":
        scan(a)
    else:
        print(json.dumps(report(ROOT / "runs" / "contamination" / a.name, settings()[1])["groups"], indent=1))


if __name__ == "__main__":
    main()
