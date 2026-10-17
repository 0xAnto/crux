"""Run a sweep on Colab T4 VMs with the `colab` CLI, unattended and restart-safe.

Run (it keeps running after the terminal or Claude Code closes; run the same command again after a reboot):
  nohup caffeinate -i .venv/bin/python -m crux.colab_sweep >> runs/step5/driver.out 2>&1 &
Show the status: cat runs/step5/sweep_status.json

The driver is the only controller. Each tick (every --tick seconds) it:
- downloads every finished run (metrics.json last, so a local metrics.json means a complete run),
- gives an idle VM the next chunk of runs (trial first), and creates a VM when one is missing,
- puts back the runs of a lost VM. A run that a VM restarts resumes from its last.pt (crux.train).
All state is in the local run folder: a restart of the driver continues where it stopped. A run that fails two
times on one VM is not given out again. When no run is left, the driver stops all VMs and runs crux.aggregate.
The VMs use the pinned commit. The driver uploads only train, dev_id, and validation (never test).
"""
import argparse
import fcntl
import gzip
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/0xAnto/crux.git"
SPLITS = ("train", "dev_id", "validation")  # never test (protocol: Step 8 only)
FILES = ("train.log", "val_probs.npz", "best.pt", "metrics.json")  # metrics.json last: it marks a complete run
VM_RUNS = "/content/runs/step5"


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def notify(msg):
    """Log the message and show a macOS notification (important events only)."""
    log("NOTIFY " + msg)
    subprocess.run(["osascript", "-e", f'display notification {json.dumps(msg)} with title "CRUX sweep"'],
                   capture_output=True)


def colab(*args, stdin=None, timeout=300):
    """Run one colab CLI command. Return (ok, output). A network error or a timeout gives ok=False."""
    try:
        r = subprocess.run(["colab", *args], input=stdin, capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout} s"


def vm_python(vm, code, timeout=120):
    """Run Python in the kernel of the VM. The code prints one line '@@<json>'. Return the object or None."""
    ok, out = colab("exec", "-s", vm, "--timeout", str(timeout), stdin=code, timeout=timeout + 120)
    for line in out.splitlines():
        if line.startswith("@@"):
            return json.loads(line[2:])
    log(f"{vm}: exec gave no result: {out[-600:]}")
    return None


def live_vms():
    """Return the names of the live sessions, or None if the CLI does not answer (network)."""
    ok, out = colab("sessions", timeout=120)
    if not ok:
        return None
    return {line[1:line.index("]")] for line in out.splitlines() if line.startswith("[") and "]" in line}


PROBE = """
import glob, json, os, subprocess, time
done = sorted(p[len('{runs}/'):-len('/metrics.json')] for p in glob.glob('{runs}/*/t*/s*/metrics.json'))
busy = subprocess.run(['pgrep', '-f', 'crux.train'], capture_output=True).returncode == 0
partial = sorted(p[len('{runs}/'):-len('/last.pt')] for p in glob.glob('{runs}/*/t*/s*/last.pt')
           if not os.path.exists(p[:-len('last.pt')] + 'metrics.json'))
failed = open('/content/failed.txt').read().split() if os.path.exists('/content/failed.txt') else []
logs = sorted(glob.glob('{runs}/*/t*/s*/train.log'), key=os.path.getmtime)
last = open(logs[-1]).read().splitlines()[-1:] if logs else []
print('@@' + json.dumps(dict(done=done, partial=partial, busy=busy, failed=failed, setup=os.path.exists('/content/setup.ok'),
                             current=logs[-1][len('{runs}/'):-len('/train.log')] if logs else None, last=last,
                             age=time.time() - os.path.getmtime(logs[-1]) if logs else 0)))
"""

SETUP = """
import json, os, subprocess
def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(c + ': ' + r.stderr[-1500:])
    return r.stdout
if not os.path.exists('/content/crux/.git'):
    sh('git clone -q {repo} /content/crux')
sh('cd /content/crux && git fetch -q origin && git checkout -q {commit} && pip install -q -r requirements.txt')
os.makedirs('/content/crux/data/splits', exist_ok=True)
print('@@' + json.dumps(True))
"""

VERIFY = """
import json, subprocess, sys
subprocess.run('cd /content/crux/data/splits && gunzip -f {names}', shell=True, check=True)
sys.path.insert(0, '/content/crux')
from crux.splits import load_split  # checks the SHA-256 against the manifest and protocol.yaml
n = {{s: len(load_split(s)) for s in {splits}}}
open('/content/setup.ok', 'w').write(json.dumps(n))
print('@@' + json.dumps(n))
"""

# The VM works through its queue file. Each run gets at most 2 attempts. A finished run is skipped.
START = """
import json, subprocess
runs = {runs_list}
open('/content/queue.txt', 'w').write('\\n'.join(runs) + '\\n')
lines = ['cd /content/crux']
for r in runs:
    arch, t, s = r.split('/')
    d = '{runs}/' + r
    lines.append(f'[ -f {{d}}/metrics.json ] || for i in 1 2; do python -m crux.train --config {config} --arch {{arch}} '
                 f'--trial {{t[1:]}} --seed-index {{s[1:]}} --out {{d}} && break; echo {{r}} >> /content/failed.txt; done')
open('/content/queue.sh', 'w').write('\\n'.join(lines) + '\\n')
subprocess.Popen('nohup bash /content/queue.sh >> /content/queue.log 2>&1 &', shell=True, start_new_session=True)
print('@@' + json.dumps(len(runs)))
"""


class Driver:
    def __init__(self, args):
        self.a = args
        self.root = ROOT / args.root
        self.state_file = self.root / "sweep_state.json"
        cfg = yaml.safe_load((ROOT / args.config).read_text())
        protocol = yaml.safe_load((ROOT / "configs" / "protocol.yaml").read_text())
        n_seeds, n_trials = len(protocol["seeds"]), len(cfg["search"]["trials"])
        # trial first, so that each finished trial has all architectures and seeds
        self.all = [f"{a}/t{t}/s{s}" for t in range(n_trials) for a in cfg["architectures"] for s in range(n_seeds)]
        self.state = json.loads(self.state_file.read_text()) if self.state_file.exists() else {
            "assigned": {}, "failed": [], "created": {}}

    def save(self, **status):
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1))
        tmp.replace(self.state_file)
        done = self.done()
        status.update(time=time.strftime("%Y-%m-%d %H:%M:%S"), done=len(done), total=len(self.all),
                      failed=self.state["failed"], assigned=self.state["assigned"])
        (self.root / "sweep_status.json").write_text(json.dumps(status, indent=1))

    def done(self):
        return {r for r in self.all if (self.root / r / "metrics.json").exists()}

    def download(self, vm, run):
        """Copy one finished run. Check the budget and the fp16 flag. Return False on a copy error."""
        dst = self.root / run
        tmp = dst.with_name(dst.name + ".part")
        tmp.mkdir(parents=True, exist_ok=True)
        for f in FILES:
            ok, out = colab("download", "-s", vm, f"{VM_RUNS}/{run}/{f}", str(tmp / f), timeout=1800)
            if not ok or not (tmp / f).exists():
                log(f"{vm}: download of {run}/{f} failed: {out[-300:]}")
                return False
        m = json.loads((tmp / "metrics.json").read_text())
        if m.get("steps") != self.a.steps or m.get("fp16") is not True:
            self.halt(f"{run}: steps {m.get('steps')} fp16 {m.get('fp16')} (expected {self.a.steps}, true)")
        dst.mkdir(parents=True, exist_ok=True)
        for f in FILES:  # metrics.json last
            (tmp / f).replace(dst / f)
        shutil.rmtree(tmp)
        log(f"{vm}: downloaded {run} (validation accuracy {m['validation'].get('accuracy', float('nan')):.4f})")
        return True

    def halt(self, reason):
        for vm in self.state["created"]:
            colab("stop", "-s", vm, timeout=300)
        self.save(halted=reason)
        notify(f"HALT: {reason}. All VMs are stopped.")
        sys.exit(1)

    def setup(self, vm):
        if vm_python(vm, SETUP.format(repo=REPO, commit=self.a.commit), timeout=1200) is None:
            return False
        cache = Path.home() / ".cache" / "crux_sweep"
        cache.mkdir(parents=True, exist_ok=True)
        for s in SPLITS:
            gz = cache / f"{s}.jsonl.gz"
            if not gz.exists():
                with open(ROOT / "data" / "splits" / f"{s}.jsonl", "rb") as src, gzip.open(gz, "wb") as out:
                    shutil.copyfileobj(src, out)
            ok, out = colab("upload", "-s", vm, str(gz), f"/content/crux/data/splits/{s}.jsonl.gz", timeout=1800)
            if not ok:
                log(f"{vm}: upload of {s} failed: {out[-300:]}")
                return False
        n = vm_python(vm, VERIFY.format(names=" ".join(f"{s}.jsonl.gz" for s in SPLITS), splits=SPLITS), 600)
        log(f"{vm}: setup {'done' if n else 'failed'}: {n}")
        return n is not None

    def tick(self):
        live = live_vms()
        if live is None:
            log("colab sessions does not answer (network or auth); try again at the next tick")
            return False
        done = self.done()
        # Runs of a lost VM go back to the pool.
        for vm in list(self.state["assigned"]):
            if vm not in live:
                lost = [r for r in self.state["assigned"].pop(vm) if r not in done]
                self.state["created"].pop(vm, None)
                if lost:
                    notify(f"{vm} is lost. {len(lost)} runs go back to the pool.")
        names = [f"{self.a.prefix}{i}" for i in range(self.a.vms)]
        info = {}
        for vm in names:
            if vm in live:
                self.state["created"].setdefault(vm, "adopted")  # a VM of an earlier controller
                p = vm_python(vm, PROBE.format(runs=VM_RUNS))
                if p is None:
                    continue
                info[vm] = p
                if p["busy"] and p.get("age", 0) > self.a.stall:  # no log line for too long: stop the run
                    vm_python(vm, "import subprocess; subprocess.run(['pkill', '-f', 'python -m crux.train']); print('@@1')")
                    notify(f"{vm}: {p['current']} wrote no log for {p['age'] / 60:.0f} min. Stopped; the queue tries "
                           f"it again from last.pt.")
                for run in p["done"]:
                    if run in self.all and run not in done and self.download(vm, run):
                        done.add(run)
                for run in set(p["failed"]):
                    if p["failed"].count(run) >= 2 and run not in self.state["failed"] and run not in done:
                        self.state["failed"].append(run)
                        notify(f"{run} failed 2 times on {vm}. See /content/queue.log on the VM.")
        if self.a.pause:  # no new run: stop each queue, let the current run end, then stop the VM
            for vm, p in info.items():
                vm_python(vm, "import subprocess; subprocess.run(['pkill', '-f', 'queue.sh']); print('@@1')")
                if not p["busy"]:
                    colab("stop", "-s", vm, timeout=300)
                    self.state["created"].pop(vm, None)
                    self.state["assigned"].pop(vm, None)
                    log(f"{vm}: paused and stopped")
            self.save(paused=True)
            return not any(vm in live for vm in self.state["created"])
        todo = [r for r in self.all if r not in done and r not in self.state["failed"]]
        if not todo:
            return True
        busy = {r for vm, rs in self.state["assigned"].items() for r in rs if r not in done}
        pool = [r for r in todo if r not in busy]
        for vm in names:
            p = info.get(vm)
            if vm in live and p is None:
                continue  # the probe did not answer; do not touch this VM now
            if p and p["busy"]:
                if vm not in self.state["assigned"]:  # a loop that the driver did not start: stop it
                    vm_python(vm, "import subprocess; subprocess.run(['pkill', '-f', 'crux.train']); print('@@1')")
                else:
                    continue
            mine = [r for r in self.state["assigned"].get(vm, []) if r in todo]
            if not mine:
                if not pool:
                    if vm in live and vm in self.state["created"]:
                        colab("stop", "-s", vm, timeout=300)
                        self.state["created"].pop(vm)
                        self.state["assigned"].pop(vm, None)
                        log(f"{vm}: no work left, stopped")
                    continue
                # First the runs that this VM started before (they resume from last.pt), then the pool order.
                own = [r for r in pool if p and r in p["partial"]]
                rest = [r for r in pool if r not in own]
                mine, pool = (own + rest)[:self.a.chunk], (own + rest)[self.a.chunk:]
            if vm not in live:
                ok, out = colab("new", "-s", vm, "--gpu", "T4", timeout=900)
                if not ok:
                    log(f"{vm}: no T4 now: {out[-200:]}")
                    pool = mine + pool
                    continue
                self.state["created"][vm] = time.strftime("%Y-%m-%d %H:%M:%S")
                live.add(vm)
                log(f"{vm}: created")
            if not (p and p["setup"]) and not self.setup(vm):
                pool = mine + pool
                continue
            if vm_python(vm, START.format(runs_list=json.dumps(mine), runs=VM_RUNS, config=self.a.config)) is None:
                pool = mine + pool
                continue
            self.state["assigned"][vm] = mine
            log(f"{vm}: started {len(mine)} runs: {mine[0]} .. {mine[-1]}")
            self.save()
        self.save(vms={vm: dict(current=p["current"], last=p["last"]) for vm, p in info.items()})
        return False

    def finish(self):
        for vm in list(self.state["created"]):
            colab("stop", "-s", vm, timeout=300)
            self.state["created"].pop(vm)
        r = subprocess.run([sys.executable, "-m", "crux.aggregate", str(self.root)], cwd=ROOT, capture_output=True,
                           text=True)
        (self.root / "aggregate.txt").write_text(r.stdout + r.stderr)
        self.save(finished=True)
        notify(f"Sweep done: {len(self.done())}/{len(self.all)} runs, {len(self.state['failed'])} failed. "
               f"See {self.root / 'aggregate.txt'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step5.yaml")
    ap.add_argument("--root", default="runs/step5")
    ap.add_argument("--commit", default="eb257e2", help="the commit that the VMs check out")
    ap.add_argument("--steps", type=int, default=6000, help="expected budget.training_steps in metrics.json")
    ap.add_argument("--prefix", default="crux5-s", help="VM names are <prefix>0, <prefix>1, ...")
    ap.add_argument("--vms", type=int, default=3, help="T4 VMs at the same time (fewer if Colab refuses)")
    ap.add_argument("--chunk", type=int, default=6, help="runs given to a VM at one time")
    ap.add_argument("--tick", type=int, default=600, help="seconds between ticks")
    ap.add_argument("--pause", action="store_true", help="start no run; stop each VM when its current run ends")
    ap.add_argument("--stall", type=int, default=1800, help="seconds without a train.log line before a run is stopped")
    args = ap.parse_args()
    d = Driver(args)
    d.root.mkdir(parents=True, exist_ok=True)
    lock = open(d.root / "driver.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # one driver at a time
    except BlockingIOError:
        sys.exit("another driver runs")
    log(f"driver starts: {len(d.done())}/{len(d.all)} runs done")
    while True:
        try:
            if d.tick():
                if args.pause:
                    notify(f"Sweep paused: {len(d.done())}/{len(d.all)} runs done, all VMs stopped.")
                    return
                d.finish()
                return
        except SystemExit:
            raise
        except Exception as e:  # a bad tick must not end the sweep; the next tick tries again
            log(f"tick error: {e!r}")
        time.sleep(args.tick)


if __name__ == "__main__":
    main()
