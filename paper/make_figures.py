"""Make the extra figure of the paper: the accuracy of each seed on test synthetic.

Input: the per-run files runs/step8/eval/<model>/s<k>/bench.json (they are not in git).
Output: paper/figures/per_seed.png
Usage: python paper/make_figures.py [runs/step8/eval] [paper/figures]
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = ["A", "A-param", "A+CF", "B", "B-abl", "B+CF", "C", "C+cache", "C+equivariant", "D", "D-CF"]
SEEDS = [0, 1, 2]
KEY = "synthetic/unseen_rule_accuracy"
MAJORITY = 0.3357  # test synthetic majority baseline (runs/step8/benchmark.txt)

root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/step8/eval")
out = Path(sys.argv[2] if len(sys.argv) > 2 else "paper/figures")
out.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(7.5, 3.6))
marks = ["o", "s", "^"]
for i, m in enumerate(MODELS):
    vals = [json.load(open(root / m / f"s{k}" / "bench.json"))["sets"]["test"]["metrics"][KEY] for k in SEEDS]
    mean = sum(vals) / len(vals)
    for k, v in enumerate(vals):
        ax.scatter(i + (k - 1) * 0.12, v, marker=marks[k], s=22, color="#1f4e8c", zorder=3)
    ax.hlines(mean, i - 0.3, i + 0.3, color="#555555", lw=1.2, zorder=2)
ax.axhline(MAJORITY, color="black", ls="--", lw=1)
ax.text(-0.45, MAJORITY + 0.008, "majority baseline", ha="left", fontsize=8)
ax.set_xticks(range(len(MODELS)))
ax.set_xticklabels(MODELS, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("unseen_rule_accuracy (higher is better)", fontsize=8)
ax.set_title("Test synthetic: one marker per seed (circle s0, square s1, triangle s2); line = mean", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(out / "per_seed.png", dpi=200)
