"""Step 8: plots of the benchmark report. Run: python -m crux.plots [runs/step8/benchmark.json]
It writes PNG files into <report folder>/plots/. Mean +- std over the seeds; the majority baseline is a dashed line.
One color per model in all plots: the hue gives the family, the lightness gives the model inside the family.
The better direction is in each title. The model names are on the axes, so no plot depends on color alone."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {  # family -> colors of its models, in the order of configs/step8_eval.yaml
    "A": ["#2a78d6", "#6da7ec", "#184f95"], "BC": ["#eb6834", "#f4a07a", "#b94a1d", "#8f3714"],
    "D": ["#1baf7a", "#5fd0a5", "#0f7f58", "#0a5c40"], "external": ["#6250d6", "#9085e9"]}
FAMILIES = {"A": "A family", "BC": "B and C family", "D": "D family", "external": "external baselines"}
SETS = {"test|synthetic": "test synthetic", "test|proofwriter/depth5": "test depth5", "test|proofwriter/birds_electricity": "test birds",
        "test|proofwriter/pararules": "test pararules", "validation|synthetic": "validation", "boardgameqa|pooled": "BoardgameQA",
        "dynabench|dynabench": "DynaBench", "supernat|pooled": "Super-NaturalInstructions"}
INK, MUTED = "#0b0b0b", "#52514e"


def model_colors(models):
    seen, out = {}, {}
    for n, m in models.items():
        i = seen[m["family"]] = seen.get(m["family"], -1) + 1
        out[n] = COLORS[m["family"]][i % len(COLORS[m["family"]])]
    return out


def _levels(res, metric):
    return {k: lv for k, lv in res["levels"].items() if k in SETS and metric in next(iter(lv["models"].values()))["metrics"]}


def _legend(fig, res, colors):
    from matplotlib.patches import Patch

    fig.legend([Patch(color=colors[n]) for n in res["models"]], [f"{n} ({FAMILIES[m['family']]})" for n, m in res["models"].items()],
               loc="lower center", ncol=4, fontsize=8, frameon=False)


def bars(res, metric, title, path, baseline=True):
    """One panel per evaluation set, one bar per model."""
    lvs, colors = _levels(res, metric), model_colors(res["models"])
    cols = min(4, len(lvs))
    fig, axes = plt.subplots(-(-len(lvs) // cols), cols, figsize=(4 * cols, 3.4 * -(-len(lvs) // cols) + 1.8), squeeze=False)
    for ax, (k, lv) in zip(axes.flat, lvs.items()):
        names = list(lv["models"])
        ax.bar(range(len(names)), [lv["models"][n]["metrics"][metric][0] for n in names], width=0.6, color=[colors[n] for n in names],
               yerr=[lv["models"][n]["metrics"][metric][1] for n in names], error_kw={"elinewidth": 1, "capsize": 2, "ecolor": MUTED})
        if baseline and lv["majority_baseline"] is not None:
            ax.axhline(lv["majority_baseline"], color=INK, ls="--", lw=1)
        ax.set_title(f"{SETS[k]} (n = {lv['n']})", fontsize=9, color=INK)
        ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7, colors=MUTED)
    for ax in list(axes.flat)[len(lvs):]:
        ax.axis("off")
    fig.suptitle(title + ("; dashed line: majority baseline" if baseline else ""), fontsize=11, color=INK)
    _legend(fig, res, colors)
    fig.tight_layout(rect=(0, 0.16, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def generalization(res, path):
    """Accuracy minus the majority baseline for each set, one dot per model."""
    lvs, colors = _levels(res, "accuracy"), model_colors(res["models"])
    fig, ax = plt.subplots(figsize=(11, 5))
    names = list(res["models"])
    for j, n in enumerate(names):
        x = [i + (j - len(names) / 2) * 0.05 for i in range(len(lvs))]
        ax.errorbar(x, [lv["models"][n]["metrics"]["accuracy"][0] - (lv["majority_baseline"] or 0) for lv in lvs.values()],
                    yerr=[lv["models"][n]["metrics"]["accuracy"][1] for lv in lvs.values()], fmt="o", ms=4, color=colors[n], elinewidth=1)
    ax.axhline(0, color=INK, ls="--", lw=1)
    ax.set_xticks(range(len(lvs)), [SETS[k] for k in lvs], fontsize=8)
    ax.set_title("Generalization: accuracy minus majority baseline (higher is better); dashed line: majority baseline", fontsize=10, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    _legend(fig, res, colors)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def latency(res, path):
    """Unseen-rule accuracy of the test synthetic block (else validation) against the L4 latency at batch size 1."""
    level = "test|synthetic" if "test|synthetic" in res["levels"] else "validation|synthetic"
    colors, lv = model_colors(res["models"]), res["levels"][level]
    metric = "unseen_rule_accuracy" if "unseen_rule_accuracy" in lv["models"]["A"]["metrics"] else "accuracy"
    fig, ax = plt.subplots(figsize=(7, 5))
    for n, m in res["models"].items():
        if m["latency_bs1_p50_ms"] is None:
            continue
        ms = lv["models"][n]["metrics"][metric]
        ax.errorbar(m["latency_bs1_p50_ms"], ms[0], yerr=ms[1], fmt="o", ms=6, color=colors[n], elinewidth=1)
        ax.annotate(n, (m["latency_bs1_p50_ms"], ms[0]), textcoords="offset points", xytext=(5, 4), fontsize=7, color=MUTED)
    ax.set_xlabel("latency p50 on the L4, batch size 1, ms (lower is better)", fontsize=8)
    ax.set_ylabel(f"{SETS[level]} {metric} (higher is better)", fontsize=8)
    ax.set_title("Latency against accuracy: left and up is better", fontsize=10, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report", nargs="?", default="runs/step8/benchmark.json")
    args = ap.parse_args()
    res = json.loads(Path(args.report).read_text())
    out = Path(args.report).parent / "plots"
    out.mkdir(exist_ok=True)
    bars(res, "accuracy", "Accuracy per evaluation set (higher is better)", out / "accuracy.png")
    bars(res, "paired_accuracy", "Counterfactual paired accuracy (higher is better)", out / "paired_accuracy.png", baseline=False)
    bars(res, "ece", "Calibration: expected calibration error (lower is better)", out / "ece.png", baseline=False)
    generalization(res, out / "generalization.png")
    latency(res, out / "latency_accuracy.png")
    print("plots in", out)


if __name__ == "__main__":
    main()
