"""crux.colab_eval against a fake colab CLI: 2 VMs share the runs, a lost VM gets them again, the baselines come once."""
import re
from types import SimpleNamespace

import yaml

from crux import colab_eval as ce


class FakeColab:
    """A VM finishes its queue at the next probe."""

    def __init__(self):
        self.vms, self.starts, self.baseline_starts = {}, [], 0

    def colab(self, *args, stdin=None, timeout=0):
        cmd, vm = args[0], (args[2] if len(args) > 2 else None)
        if cmd == "new":
            self.vms[vm] = {"have": set(), "done": [], "queue": [], "setup": False, "base": False, "baseline_job": False}
        elif cmd == "stop":
            self.vms.pop(vm, None)
        elif cmd == "upload" and args[4].endswith("best.pt"):
            self.vms[vm]["have"].add("/".join(args[4].split("/")[-4:-3] + args[4].split("/")[-2:-1]))
        elif cmd == "download":
            open(args[4], "w").write('{"complete": true, "fact_count": {}}' if args[3].endswith(".json") else "x")
        return True, ""

    def vm_python(self, vm, code, timeout=0):
        v = self.vms.get(vm)
        if v is None:
            return None
        if "pgrep" in code:
            v["done"] += v["queue"]
            v["queue"], v["base"] = [], v["base"] or v["baseline_job"]
            return dict(done=v["done"], have=sorted(v["have"]), baselines=v["base"], busy=bool(v["queue"]), setup=v["setup"], last=[])
        if "gunzip" in code:
            v["setup"] = True
        if "crux.benchmark" in code:
            v["queue"] = re.findall(r"'([^']+)'", code.split("--runs ")[1].split(" >> ")[0])
            v["baseline_job"] = "--baselines" in code
            self.starts.append((vm, list(v["queue"]), v["baseline_job"]))
        return True


def test_two_vms_share_the_runs_and_a_lost_vm_is_replaced(tmp_path, monkeypatch):
    fake = FakeColab()
    for name in ("colab", "vm_python"):
        monkeypatch.setattr(ce, name, getattr(fake, name))
    monkeypatch.setattr(ce, "live_vms", lambda: set(fake.vms))
    monkeypatch.setattr(ce, "log", lambda m: None)
    monkeypatch.setattr(ce, "notify", lambda m: None)
    monkeypatch.setattr(ce.Driver, "setup", lambda self, vm, th: fake.vms[vm].update(setup=True) or True)
    models = {m: {"root": str(tmp_path / "runs"), "family": "A"} for m in ("A", "B", "C")}
    for m in models:
        for k in range(3):
            d = tmp_path / "runs" / m / "t0" / f"s{k}"
            d.mkdir(parents=True)
            (d / "best.pt").write_bytes(b"x" * (10 + k))
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump({"out_root": str(tmp_path / "out"), "sets": ["test"], "models": models}))
    driver = ce.Driver(SimpleNamespace(config=str(tmp_path / "cfg.yaml"), vms=2, prefix="vm", chunk=4, commit="c", max_starts=3))
    assert sorted(driver.parts[0] + driver.parts[1]) == sorted(f"{m}/s{k}" for m in "ABC" for k in range(3))
    finished, lost = False, False
    for _ in range(12):
        finished = driver.tick()
        if finished:
            break
        if not lost and fake.starts and "vm1" in fake.vms:
            fake.vms.pop("vm1")  # a lost VM: its runs are given again
            lost = True
    assert finished and all(driver.complete(r) for p in driver.parts for r in p) and not driver.need_baselines()
    assert sum(b for _, _, b in fake.starts) == 1 and all(vm == "vm0" for vm, _, b in fake.starts if b)  # the baselines: VM 0, once
    assert all(len(q) <= 4 for _, q, _ in fake.starts), fake.starts  # chunks of 4
    assert not fake.vms, fake.vms  # all VMs stopped
