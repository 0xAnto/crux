# CRUX paper

This folder holds the LaTeX source of the CRUX paper.

## Files

- `main.tex`: the paper (article class, no venue style file).
- `refs.bib`: the bibliography. 
- `figures/`: the figures. `accuracy.png`, `paired_accuracy.png`, `ece.png`, and `latency_accuracy.png` are copies of `runs/step8/plots/`. `per_seed.png` comes from `make_figures.py`.
- `make_figures.py`: makes `figures/per_seed.png` from `runs/step8/eval/<model>/s<k>/bench.json`.

## Build

Use one of these commands in this folder.

```
latexmk -pdf main.tex
tectonic main.tex
```

The packages are `booktabs`, `array`, `graphicx`, `amsmath`, `amssymb`, `xcolor`, `hyperref`, `microtype`, `lmodern`, and `geometry`.
BibTeX runs with the `plain` style. `tectonic` runs it by itself.

## Make the extra figure again

The run outputs are not in git. Run the command from the repository root, in an environment with `matplotlib`.

```
python paper/make_figures.py runs/step8/eval paper/figures
```

## Placeholders to fill before release

- `\REPRO{}` in Section "Reproducibility": the result of the reproduction check (A seed 0 and D seed 0, clean Colab VM, commit `2d3e4b0`). It prints in red until you replace it.

The PDF `main.pdf` is a build output. It is not tracked in git.
