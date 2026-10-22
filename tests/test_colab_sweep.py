"""crux.colab_sweep against a fake colab CLI: all runs arrive once, a lost VM gives its runs back, test is never sent."""
import json
from types import SimpleNamespace

import pytest

from crux import colab_sweep as cs


class FakeColab:
    """VMs finish one queued run per probe. Only `max_vms` T4 VMs can exist at the same time."""

    def __init__(self, max_vms):
        self.max_vms, self.vms, self.uploads, self.started = max_vms, {}, [], []

    def colab(self, *args, stdin=None, timeout=0):
        cmd, vm = args[0], (args[2] if len(args) > 2 else None)
        if cmd == "sessions":
            return True, "".join(f"[{v}] x | Hardware: T4\n" for v in self.vms)
        if cmd == "new":
            if len(self.vms) >= self.max_vms:
                return False, "quota"
            self.vms[vm] = {"queue": [], "done": [], "setup": False}
            return True, ""
        if cmd == "stop":
            self.vms.pop(vm, None)
            return True, ""
        if cmd == "upload":
            self.uploads.append(args[3])
            return True, ""
        if cmd == "download":
            run = args[3][len(cs.VM_RUNS) + 1:].rsplit("/", 1)[0]
            assert run in self.vms[vm]["done"]
            m = {"steps": 6000, "fp16": True, "validation": {"accuracy": 0.5}}
            open(args[4], "w").write(json.dumps(m))
            return True, ""
        raise AssertionError(args)

    def vm_python(self, vm, code, timeout=0):
        v = self.vms.get(vm)
        if v is None:
            return None
        if "pgrep" in code:  # probe: the VM finishes its next run
            if v["queue"]:
                v["done"].append(v["queue"].pop(0))
            return dict(done=v["done"], partial=[], busy=bool(v["queue"]), failed=[], setup=v["setup"],
                        current=None, last=[], age=v.get("age", 0))
        if "queue.sh']" in code:  # pause: the queue stops, the current run continues
            v["queue"] = v["queue"][:1]
            return 1
        if "pkill" in code:  # the stalled run stops; the queue tries it again
            v["killed"] = v.get("killed", 0) + 1
            v["age"] = 0
            return 1
        if "git clone" in code:
            return True
        if "load_split" in code:
            v["setup"] = True
            return {}
        if "queue.sh" in code:
            runs = json.loads(code.split("runs = ", 1)[1].split("\n", 1)[0])
            v["queue"] = list(runs)
            self.started += runs
            return len(runs)
        raise AssertionError(code)


def make(tmp_path, monkeypatch, fake, vms=3):
    monkeypatch.setattr(cs, "colab", fake.colab)
    monkeypatch.setattr(cs, "vm_python", fake.vm_python)
    monkeypatch.setattr(cs, "live_vms", lambda: set(fake.vms))
    monkeypatch.setattr(cs, "notify", lambda msg: None)
    monkeypatch.setattr(cs, "ROOT", tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "protocol.yaml").write_text("seeds: [1, 2, 3]\n")
    (tmp_path / "configs" / "s.yaml").write_text("architectures: {A: {}, B: {}}\nsearch: {trials: [{}, {}]}\n")
    (tmp_path / "data" / "splits").mkdir(parents=True)
    for s in cs.SPLITS:
        (tmp_path / "data" / "splits" / f"{s}.jsonl").write_text("{}\n")
    monkeypatch.setattr(cs.Path, "home", lambda: tmp_path)
    args = SimpleNamespace(config="configs/s.yaml", root="runs", commit="x", steps=6000, prefix="vm", vms=vms,
                           chunk=3, stall=1800, pause=False, archs=None, seeds=None)
    d = cs.Driver(args)
    d.root.mkdir()
    return d


def test_all_runs_arrive_with_a_lost_vm_and_a_refused_t4(tmp_path, monkeypatch):
    fake = FakeColab(max_vms=2)
    d = make(tmp_path, monkeypatch, fake)
    lost = False
    for _ in range(100):
        if d.tick():
            break
        if not lost and fake.vms.get("vm0", {}).get("queue"):
            fake.vms.pop("vm0")  # the VM disappears with queued runs
            lost = True
    else:
        pytest.fail("the sweep did not finish")
    assert lost
    assert d.done() == set(d.all) and len(d.all) == 12  # 2 trials x 2 archs x 3 seeds
    assert all(not u.split("/")[-1].startswith("test") for u in fake.uploads)
    assert len(fake.started) > len(d.all)  # the runs of the lost VM were given out again
    d.finish = lambda: None
    assert not any(v["queue"] for v in fake.vms.values())


def test_wrong_budget_halts_and_stops_vms(tmp_path, monkeypatch):
    fake = FakeColab(max_vms=1)
    d = make(tmp_path, monkeypatch, fake, vms=1)
    orig = fake.colab

    def bad(*args, **kw):
        ok, out = orig(*args, **kw)
        if args[0] == "download":
            open(args[4], "w").write(json.dumps({"steps": 50, "fp16": True, "validation": {}}))
        return ok, out

    monkeypatch.setattr(cs, "colab", bad)
    with pytest.raises(SystemExit):
        for _ in range(10):
            d.tick()
    assert not fake.vms
    assert "halted" in json.loads((d.root / "sweep_status.json").read_text())


def test_a_stalled_run_is_stopped(tmp_path, monkeypatch):
    fake = FakeColab(max_vms=1)
    d = make(tmp_path, monkeypatch, fake, vms=1)
    d.tick()  # creates vm0 and starts its queue
    fake.vms["vm0"]["queue"].append("A/t1/s0")  # keep it busy
    fake.vms["vm0"]["age"] = 1000
    d.tick()
    assert fake.vms["vm0"].get("killed", 0) == 0  # a quiet run below the limit is not touched
    fake.vms["vm0"]["age"] = 4000
    d.tick()
    assert fake.vms["vm0"]["killed"] == 1


def test_pause_finishes_the_current_run_then_stops_all_vms(tmp_path, monkeypatch):
    fake = FakeColab(max_vms=2)
    d = make(tmp_path, monkeypatch, fake, vms=2)
    d.tick()
    d.tick()
    n = len(d.done())
    d.a.pause = True
    for _ in range(5):
        if d.tick():
            break
    assert not fake.vms
    assert n < len(d.done()) < len(d.all)  # the current runs arrived, the queued runs did not start


def test_archs_and_seeds_select_the_runs(tmp_path, monkeypatch):
    d = make(tmp_path, monkeypatch, FakeColab(max_vms=1))
    d.a.archs, d.a.seeds = ["B"], [0]
    assert cs.Driver(d.a).all == ["B/t0/s0", "B/t1/s0"]
