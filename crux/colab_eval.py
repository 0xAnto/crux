"""Step 8: run the whole benchmark (crux.benchmark --final) on 2 Colab L4 VMs, unattended and resumable.

Run: nohup caffeinate -i .venv/bin/python -m crux.colab_eval --commit <pinned commit> >> runs/step8/eval_driver.out 2>&1 &
Plan only: python -m crux.colab_eval --dry-run
The runs (best.pt of each model and seed in configs/step8_eval.yaml) go to the VMs in 2 parts of about the same size.
VM 0 also computes the baselines (majority and fact-count probe) in its last chunk. A VM gets a chunk of runs, starts
`crux.benchmark --final` for it, and the driver downloads each run as it finishes. The driver keeps no state:
every tick it compares the local outputs with the plan, so a restart of the driver or a lost VM (new VM, new upload) resumes.
A run is complete when its bench.json has "complete": true; the driver downloads bench.json last.
At the end it stops the VMs and writes the report: crux.aggregate --bench, then crux.plots.
The VMs use the pinned commit. The test split goes to the VMs, but only the final run reads it (the guard in crux.benchmark).
"""
import argparse
import gzip
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from crux.colab_sweep import REPO, colab, live_vms, log, notify, vm_python
from crux.splits import SPLITS

ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILES = [f"{s}.jsonl" for s in SPLITS] + [f"benchmarks/{b}.jsonl" for b in ("boardgameqa", "dynabench", "supernat")]
THEORIES = "data/processed/synthetic/main/theories.jsonl"  # for the fact-count probe (VM 0)
VM_OUT = "/content/eval"

PROBE = """
import glob, json, os, subprocess
done = [p[len('{out}/'):-len('/bench.json')] for p in glob.glob('{out}/*/s*/bench.json') if json.load(open(p)).get('complete')]
have = ['/'.join([p.split('/')[-4], p.split('/')[-2]]) for p in glob.glob('/content/crux/runs/*/*/t0/s*/best.pt')]
base = os.path.exists('{out}/baselines.json') and 'fact_count' in json.load(open('{out}/baselines.json'))
busy = subprocess.run(['pgrep', '-f', 'crux.benchmark'], capture_output=True).returncode == 0
log = open('/content/bench.log').read().splitlines()[-2:] if os.path.exists('/content/bench.log') else []
print('@@' + json.dumps(dict(done=done, have=have, baselines=base, busy=busy, setup=os.path.exists('/content/setup.ok'), last=log)))
"""

SETUP = """
import json, os, subprocess
def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(c + ': ' + r.stderr[-1500:])
if not os.path.exists('/content/crux/.git'):
    sh('git clone -q {repo} /content/crux')
sh('cd /content/crux && git fetch -q origin && git checkout -q {commit} && pip install -q -r requirements.txt')
os.makedirs('/content/crux/data/splits/benchmarks', exist_ok=True)
os.makedirs('/content/crux/data/processed/synthetic/main', exist_ok=True)
print('@@' + json.dumps(True))
"""

VERIFY = """
import json, subprocess
subprocess.run('cd /content/crux/data && gunzip -f $(find . -name "*.gz")', shell=True, check=True)
r = subprocess.run('cd /content/crux && python -m crux.splits verify', shell=True, capture_output=True, text=True)  # hashes only: no parse
if r.returncode:
    raise SystemExit(r.stdout + r.stderr)
open('/content/setup.ok', 'w').write('ok')
print('@@' + json.dumps(r.stdout))
"""

START = """
import subprocess, json
subprocess.Popen({cmd!r}, shell=True, start_new_session=True)
print('@@' + json.dumps(True))
"""

MKDIR = """
import json, os
for d in {dirs}:
    os.makedirs(d, exist_ok=True)
print('@@' + json.dumps(True))
"""


def plan(cfg, n_vms):
    """All runs with a best.pt, split into n_vms parts of about the same checkpoint size (greedy, largest first)."""
    n_seeds = len(yaml.safe_load((ROOT / "configs" / "protocol.yaml").read_text())["seeds"])
    runs = [(m, k) for m in cfg["models"] for k in range(n_seeds) if run_dir(cfg, m, k, "best.pt").exists()]
    parts, sizes = [[] for _ in range(n_vms)], [0] * n_vms
    for m, k in sorted(runs, key=lambda r: -run_dir(cfg, *r, "best.pt").stat().st_size):
        i = sizes.index(min(sizes))
        parts[i].append(f"{m}/s{k}")
        sizes[i] += run_dir(cfg, m, k, "best.pt").stat().st_size
    return [sorted(p) for p in parts], sizes


def run_dir(cfg, model, k, name=""):
    return ROOT / cfg["models"][model]["root"] / model / "t0" / f"s{k}" / name


def split_run(r):
    m, k = r.rsplit("/s", 1)
    return m, int(k)


class Driver:
    def __init__(self, a):
        self.a = a
        self.cfg = yaml.safe_load((ROOT / a.config).read_text())
        self.out = ROOT / self.cfg["out_root"]
        self.parts, self.sizes = plan(self.cfg, a.vms)
        self.names = [f"{a.prefix}{i}" for i in range(a.vms)]
        self.starts = {}  # vm -> (the runs left at the last start, number of starts with the same runs)

    def complete(self, r):
        f = self.out / r / "bench.json"
        return f.exists() and json.loads(f.read_text()).get("complete", False)

    def need_baselines(self):
        f = self.out / "baselines.json"
        return not (f.exists() and "fact_count" in json.loads(f.read_text()))

    def upload(self, vm, src, dst):
        ok, out = colab("upload", "-s", vm, str(src), dst, timeout=3600)
        if not ok:
            log(f"{vm}: upload of {src.name} failed: {out[-300:]}")
        return ok

    def setup(self, vm, with_theories):
        if vm_python(vm, SETUP.format(repo=REPO, commit=self.a.commit), timeout=300) is None:
            return False
        cache = Path.home() / ".cache" / "crux_eval"
        for f in SPLIT_FILES + ([THEORIES] if with_theories else []):
            src = ROOT / (THEORIES if f == THEORIES else f"data/splits/{f}")
            gz = cache / (f.replace("/", "_") + ".gz")
            gz.parent.mkdir(parents=True, exist_ok=True)
            if not gz.exists():
                with open(src, "rb") as s, gzip.open(gz, "wb") as o:
                    shutil.copyfileobj(s, o)
            if not self.upload(vm, gz, f"/content/crux/{THEORIES if f == THEORIES else 'data/splits/' + f}.gz"):
                return False
        ok = vm_python(vm, VERIFY, 600) is not None
        log(f"{vm}: setup {'done' if ok else 'failed'}")
        return ok

    def download(self, vm, r):
        m, k = split_run(r)
        dst = self.out / m / f"s{k}"
        tmp = dst.with_name(dst.name + ".part")
        tmp.mkdir(parents=True, exist_ok=True)
        sets = self.cfg["sets"]
        for f in [f"probs_{s}.npz" for s in sets] + ["bench.json"]:  # bench.json last
            ok, out = colab("download", "-s", vm, f"{VM_OUT}/{r}/{f}", str(tmp / f), timeout=1800)
            if not ok or not (tmp / f).exists():
                log(f"{vm}: download of {r}/{f} failed: {out[-300:]}")
                return False
        dst.mkdir(parents=True, exist_ok=True)
        for f in [f"probs_{s}.npz" for s in sets] + ["bench.json"]:
            (tmp / f).replace(dst / f)
        shutil.rmtree(tmp)
        log(f"{vm}: downloaded {r}")
        return True

    def tick(self):
        live = live_vms()
        if live is None:
            log("colab sessions does not answer; try again at the next tick")
            return False
        for i, vm in enumerate(self.names):
            todo = [r for r in self.parts[i] if not self.complete(r)]
            base = i == 0 and self.need_baselines()
            p = vm_python(vm, PROBE.format(out=VM_OUT)) if vm in live else None
            if p:
                for r in p["done"]:
                    if r in self.parts[i] and not self.complete(r):
                        self.download(vm, r)
                if i == 0 and p["baselines"] and base:
                    ok, _ = colab("download", "-s", vm, f"{VM_OUT}/baselines.json", str(self.out / "baselines.json"), timeout=600)
                    base = base and not ok
                todo = [r for r in todo if not self.complete(r)]
            if not todo and not base:
                if vm in live:
                    colab("stop", "-s", vm, timeout=300)
                    log(f"{vm}: no work left, stopped")
                continue
            if vm in live and p is None or (p and p["busy"]):
                continue  # no answer, or the job runs
            if vm not in live:
                ok, out = colab("new", "-s", vm, "--gpu", "L4", timeout=900)
                if not ok:
                    log(f"{vm}: no L4 now: {out[-200:]}")
                    continue
                notify(f"{vm} is created ({len(todo)} runs)")
                p = None
            if not (p and p["setup"]) and not self.setup(vm, i == 0):
                continue
            chunk = todo[:self.a.chunk]
            new = [split_run(r) for r in chunk if p is None or r not in p["have"]]  # the checkpoints of the chunk
            dirs = {(m, k): f"/content/crux/{self.cfg['models'][m]['root']}/{m}/t0/s{k}" for m, k in new}
            if new and vm_python(vm, MKDIR.format(dirs=json.dumps(sorted(dirs.values())))) is None:
                continue  # colab upload does not make the parent folder
            if not all(self.upload(vm, run_dir(self.cfg, m, k, f), f"{d}/{f}") for (m, k), d in dirs.items()
                       for f in ("resolved_config.yaml", "best.pt") if run_dir(self.cfg, m, k, f).exists()):
                continue  # start no job with a missing checkpoint; the next tick tries again
            left, n = self.starts.get(vm, (None, 0))
            self.starts[vm] = (todo, n + 1 if left == todo else 1)
            if self.starts[vm][1] > self.a.max_starts:  # the job stops without progress: a bug or an OOM, not a lost VM
                for v in self.names:
                    colab("stop", "-s", v, timeout=300)
                notify(f"HALT: {vm} started the same job {self.starts[vm][1] - 1} times without progress. See /content/bench.log.")
                sys.exit(1)
            extra = "--baselines" if base and len(chunk) == len(todo) else ""  # the baselines come with the last chunk of VM 0
            cmd = (f"cd /content/crux && nohup python -m crux.benchmark --final --out {VM_OUT} "
                   f"--runs {' '.join(map(shlex.quote, chunk))} {extra} >> /content/bench.log 2>&1 &")
            if vm_python(vm, START.format(cmd=cmd)) is not None:
                log(f"{vm}: started {chunk} {extra}")
        return all(self.complete(r) for part in self.parts for r in part) and not self.need_baselines()

    def finish(self):
        for vm in self.names:
            colab("stop", "-s", vm, timeout=300)
        for cmd in (["-m", "crux.aggregate", "--bench", self.a.config], ["-m", "crux.plots"]):
            r = subprocess.run([sys.executable, *cmd], cwd=ROOT, capture_output=True, text=True)
            log(f"{' '.join(cmd)}: exit {r.returncode} {r.stderr[-300:] if r.returncode else ''}")
        notify("Step 8 benchmark done. See runs/step8/benchmark.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step8_eval.yaml")
    ap.add_argument("--commit", help="the commit that the VMs check out (default: HEAD; the tree must be clean)")
    ap.add_argument("--vms", type=int, default=2, help="Colab refuses a third VM")
    ap.add_argument("--prefix", default="crux8-s")
    ap.add_argument("--chunk", type=int, default=6, help="runs given to a VM in one job")
    ap.add_argument("--tick", type=int, default=300, help="seconds between ticks")
    ap.add_argument("--max-starts", type=int, default=3, help="halt after this many starts of a job without progress")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    args = ap.parse_args()
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    args.commit = args.commit or git("rev-parse", "HEAD")
    d = Driver(args)
    if args.dry_run:
        for i, (part, size) in enumerate(zip(d.parts, d.sizes)):
            print(f"{d.names[i]}: {len(part)} runs, {size / 1e6:.0f} MB of checkpoints" + (", baselines" if i == 0 else ""))
            print("  " + " ".join(part))
        print(f"commit {args.commit}; splits and benchmarks (gzip) go to each VM; sets {d.cfg['sets']}; out {d.out}")
        return
    if git("status", "--porcelain", "--untracked-files=no") or not git("branch", "-r", "--contains", args.commit):
        sys.exit("the working tree is not clean, or the commit is not pushed: the VMs would run other code")
    while True:
        try:
            if d.tick():
                return d.finish()
        except Exception as e:  # a bad tick must not end the run; the next tick tries again
            log(f"tick error: {e!r}")
        time.sleep(args.tick)


if __name__ == "__main__":
    main()
