import hashlib
import json

import pytest
import yaml

from crux import splits as S


def rec(id_, state, instruction="Question: is it?", unit=None, **meta):
    return {"id": id_, "source": "synthetic", "task_type": "binary", "state": state, "instruction": instruction,
            "options": ["True", "False"], "target": 0, "meta": {"unit_id": unit or id_, **meta}}


def splits(**given):
    return {name: given.get(name, []) for name in S.SPLITS + S.BENCHMARKS}


def test_dev_id_keeps_every_unit_whole_and_is_deterministic():
    records = [rec(f"u{u}-{k}", "x", unit=f"u{u}") for u in range(400) for k in range(3)]
    train, dev = S.split_train_dev(records, seed=1, share=0.05)
    assert {r["meta"]["unit_id"] for r in train}.isdisjoint(r["meta"]["unit_id"] for r in dev)
    assert len(dev) % 3 == 0 and 0 < len(dev) < 0.15 * len(records)
    assert (train, dev) == S.split_train_dev(list(records), seed=1, share=0.05)
    assert dev != S.split_train_dev(records, seed=2, share=0.05)[1]


def test_id_check_finds_a_shared_template_group_but_not_a_clean_split():
    clean = splits(train=[rec("a", "x", template_id="t0")], validation=[rec("b", "y", template_id="t1")])
    S.check_ids(clean)
    clean["test"] = [rec("c", "z", template_id="t0")]
    with pytest.raises(ValueError, match="train~test:template_id"):
        S.check_ids(clean)


def test_id_check_covers_dev_id_against_every_other_split():
    ok = splits(train=[rec("a", "x", template_id="t0", unit="u0")], dev_id=[rec("d", "x", template_id="t0", unit="u1")],
                validation=[rec("v", "y", template_id="t1", unit="u2")], test=[rec("c", "z", template_id="t2", unit="u3")])
    S.check_ids(ok)  # dev_id shares the template with train and nothing with validation and test
    for name, kw, message in (("train", {"unit": "u1"}, "train~dev_id:unit_id"), ("validation", {"template_id": "t0"}, "dev_id~validation:template_id"),
                              ("test", {"template_id": "t0"}, "dev_id~test:template_id"), ("test", {"unit": "u1"}, "dev_id~test:unit_id")):
        bad = {**ok, name: ok[name] + [rec("n", "w", **{"template_id": "t9", **kw})]}
        with pytest.raises(ValueError, match=message):
            S.check_ids(bad)


def test_near_duplicate_removes_the_whole_unit_and_the_report_counts_it():
    train = [rec("t1", "the cat sat on the mat today and the dog barked loudly at night")]
    copy = rec("v1-a", "the cat sat on the mat today and the dog barked loudly at night, so he hid", unit="v1")
    mate = rec("v1-b", "bees hum softly above red tulips near a quiet pond", unit="v1")
    other = rec("v2-a", "nine green frogs jump over seven wet stones by the lake", unit="v2")
    clean, report = S.leakage(splits(train=train, validation=[copy, mate, other]), n=3, max_overlap=0.1)
    assert [r["id"] for r in clean["validation"]] == ["v2-a"]
    assert report["removed"]["validation"]["ids"] == ["v1-a", "v1-b"]
    assert report["ngram"]["pairs"]["validation~train"]["over"] == 1
    assert report["ngram"]["pairs"]["validation~train"]["max"] > 0.5


def test_exact_duplicate_is_removed_from_the_later_split():
    a = rec("a", "same state", "same rules. Question: q?")
    b = rec("b", "same state", "same rules. Question: q?")
    clean, _ = S.leakage(splits(validation=[a], test=[b]), n=8, max_overlap=0.1)
    assert [len(clean["validation"]), len(clean["test"])] == [1, 0]


def test_output_bytes_do_not_depend_on_the_input_order(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "ROOT", tmp_path)
    records = [rec(f"r{i}", f"state {i}") for i in range(20)]
    first = S.write_splits(splits(train=records), tmp_path / "a")
    second = S.write_splits(splits(train=records[::-1]), tmp_path / "b")
    assert first["train"]["sha256"] == second["train"]["sha256"]


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "ROOT", tmp_path)
    monkeypatch.setattr(S, "MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setattr(S, "PROTOCOL", tmp_path / "protocol.yaml")
    entries = S.write_splits(splits(train=[rec("a", "x")], test=[rec("b", "y")]), tmp_path / "out")
    S.MANIFEST.write_text(json.dumps({"splits": entries}))
    S.PROTOCOL.write_text(yaml.safe_dump({"splits": {"split_hash": {k: entries[k]["sha256"] for k in ("train", "test")}}}))
    return tmp_path


def test_load_split_checks_the_hash_and_guards_the_test_split(frozen):
    assert [r["id"] for r in S.load_split("train")] == ["a"]
    with pytest.raises(PermissionError):
        S.load_split("test")
    assert [r["id"] for r in S.load_split("test", allow_test=True)] == ["b"]
    path = frozen / "out" / "train.jsonl"
    path.write_text(path.read_text().replace('"x"', '"changed"'))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        S.load_split("train")
    assert hashlib.sha256(path.read_bytes()).hexdigest() != json.loads(S.MANIFEST.read_text())["splits"]["train"]["sha256"]


def test_load_split_checks_the_hash_pre_registered_in_the_protocol(frozen):
    protocol = yaml.safe_load(S.PROTOCOL.read_text())
    protocol["splits"]["split_hash"]["train"] = "0" * 64  # the manifest and the file agree, the protocol does not
    S.PROTOCOL.write_text(yaml.safe_dump(protocol))
    with pytest.raises(ValueError, match="protocol.yaml"):
        S.load_split("train")
    assert [r["id"] for r in S.load_split("test", allow_test=True)] == ["b"]


def test_the_test_entry_has_no_label_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "ROOT", tmp_path)
    entries = S.write_splits(splits(train=[rec("a", "x")], test=[rec("b", "y")]), tmp_path / "out")
    assert "label" in entries["train"] and "label" not in entries["test"] and entries["test"]["n"] == 1


def test_refused_build_keeps_the_frozen_files_and_the_manifest(tmp_path, monkeypatch):
    for name, value in (("ROOT", tmp_path), ("MANIFEST", tmp_path / "manifest.json"), ("CONFIG", tmp_path / "config.yaml")):
        monkeypatch.setattr(S, name, value)
    S.CONFIG.write_text(yaml.safe_dump({"seed": 1, "dev_share": 0.1, "ngram_n": 8, "max_overlap": 0.1, "supernat_cap": 1,
                                        "boardgameqa_splits": ["test"], "out_dir": "out", "contamination_removed": "removed.txt"}))
    (tmp_path / "removed.txt").write_text("# none\n")
    state = {"text": "first"}
    monkeypatch.setattr(S, "collect", lambda cfg: (splits(train=[rec("a", state["text"])], test=[rec("b", "y")]), {}))
    monkeypatch.setattr(S, "baselines", lambda *args: {})
    S.build()
    train, manifest = tmp_path / "out" / "train.jsonl", S.MANIFEST.read_text()
    first = train.read_bytes()
    S.build()  # the same input: the build passes
    state["text"] = "second"
    with pytest.raises(SystemExit, match="differ"):
        S.build()
    assert train.read_bytes() == first and S.MANIFEST.read_text() == manifest and not (tmp_path / "out.new").exists()
    S.build(refreeze=True)
    assert train.read_bytes() != first and S.MANIFEST.read_text() != manifest


def test_baselines_have_no_test_statistics_and_use_the_train_majority(monkeypatch):
    monkeypatch.setattr(S, "fact_count", lambda *args: {})
    meta = dict(composition_id="c0", pair_role="orig", label_flip=False, edit_relevant=False, edit_type="x")

    def syn(i, target, **kw):
        return {**rec(i, "s"), "options": ["True", "False", "Unknown"], "target": target, "meta": {"unit_id": i, **meta, **kw}}
    train = [syn("t1", 2), syn("t2", 2), syn("t3", 0)]  # the train majority is Unknown
    val = [syn("v1", 0, composition_id="c9"), syn("v2", 0, composition_id="c9"), syn("v3", 2, composition_id="c9")]
    out = S.baselines(splits(train=train, validation=val, test=[{**rec("x", "s"), "meta": {}}]), {})
    assert out["majority"]["validation"]["synthetic/binary"] == {"n": 3, "majority": round(1 / 3, 4)}  # its own majority is True
    assert out["composition"]["validation"]["c9/orig"]["majority"] == round(1 / 3, 4)
    assert all("test" not in out[k] for k in ("majority", "composition", "pairs", "fact_count"))
