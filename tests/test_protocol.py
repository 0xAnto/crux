import json
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]
P = yaml.safe_load((ROOT / "configs" / "protocol.yaml").read_text())

CONFIGS = {"A", "B", "C", "D", "A-param", "A+CF", "B+CF", "C+cache", "C+equivariant",
           "Ettin-32M", "DeBERTa-v3-xsmall"}


def test_three_unique_integer_seeds():
    seeds = P["seeds"]
    assert len(seeds) == 3 and len(set(seeds)) == 3
    assert all(isinstance(s, int) for s in seeds)


def test_bootstrap_ci_level():
    assert P["statistics"]["bootstrap"]["ci_level"] == 0.95


def test_test_split_is_evaluated_once():
    assert P["splits"]["test"]["evaluate_once"] is True


def test_all_configurations_listed():
    listed = {c["id"] for group in P["configurations"].values() for c in group}
    assert CONFIGS <= listed


def test_mobile_metrics_present():
    need = {"model_size", "peak_ram", "latency_p50", "latency_p95", "int8_accuracy_loss"}
    assert need <= set(P["mobile"]["metrics"])
    assert P["mobile"]["device"] not in (None, "", "TBD")
    assert P["mobile"]["measurement"]["batch_size"] == 1


def test_amendments_start_empty():
    assert P["amendments"] == []


def test_step4_fields_are_filled_and_match_the_manifest():
    manifest = json.loads((ROOT / "configs" / "splits_manifest.json").read_text())
    settings = yaml.safe_load((ROOT / "configs" / "step4_splits.yaml").read_text())
    assert P["splits"]["leakage"]["ngram_n"] == settings["ngram_n"] == manifest["settings"]["ngram_n"]
    assert P["splits"]["leakage"]["max_overlap"] == settings["max_overlap"]
    for name, key in (("train", "train"), ("validation", "validation"), ("test", "test")):
        assert P["splits"][name]["size"] == manifest["splits"][name]["n"] > 0
        assert P["splits"]["split_hash"][key] == manifest["splits"][name]["sha256"]
