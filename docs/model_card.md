---
license: apache-2.0
base_model: jhu-clsp/ettin-encoder-17m
library_name: pytorch
datasets:
- 0xanto/crux-data
language:
- en
tags:
- rule-reasoning
- counterfactual
- negative-result
---

# CRUX model card

## Summary

CRUX is a research project.
It tests if rule-conditioned computation and counterfactual learning help a small neural network (about 17M parameters) to decide tasks with rules that the network did not see in training.
The release has 11 models with 3 seeds each (33 sets of weights).
All models start from `jhu-clsp/ettin-encoder-17m` (MIT). See `NOTICE`.

| Model | Content |
|---|---|
| A | The Ettin-17M encoder with a standard decision head (the baseline) |
| A-param | A with the same trainable parameters as B, without conditioning |
| A+CF | A trained on the paired counterfactual data |
| B | A with an instruction-conditioned low-rank hypernetwork (Rule Compiler), head only |
| B-abl | B with hypernetwork adapters for the top 2 encoder layers |
| B+CF | B trained on the paired counterfactual data |
| C | B+CF with a paraphrase consistency loss |
| C+cache | C with independently cacheable state and rule representations |
| C+equivariant | C with permutation-equivariant candidate scoring |
| D | C with both parts of C+cache and C+equivariant |
| D-CF | D without the paired data and without the consistency loss (a control) |

## Intended use

Research only.
Use the weights to repeat or to check the experiments of the project.
Do not use the models for a real decision. They are not reliable on the data of this project (see the results).

## Files

`weights/<model>/s<k>/model.safetensors` holds the model state dict of the best validation checkpoint. It holds no optimizer state.
`weights/<model>/s<k>/config.yaml` holds the config of the run.
To load a model, call `crux.models.build_model(config["model"])`, then call `load_state_dict` with `safetensors.torch.load_file`.
The code is at https://github.com/0xAnto/crux. The data is at https://huggingface.co/datasets/0xanto/crux-data.
`MANIFEST.json` holds the SHA-256 of every file and the hashes of the splits.

## Results

The project read the frozen test split one time (Step 8).
All values are the mean over 3 seeds.
The majority baseline of the test synthetic subset uses the train label as the reference.

Accuracy (higher is better):

| Model | test synthetic | BoardgameQA | DynaBench | Super-NaturalInstructions |
|---|---|---|---|---|
| majority baseline | 0.336 | 0.367 | 0.508 | 0.442 |
| A | 0.640 | 0.373 | 0.499 | 0.440 |
| B | 0.636 | 0.367 | 0.475 | 0.434 |
| C | 0.513 | 0.368 | 0.534 | 0.441 |
| D | 0.442 | 0.368 | 0.510 | 0.433 |

Counterfactual paired accuracy on test synthetic (higher is better): A 0.481, B 0.472, C 0.327, D 0.241.

The negative result:

- No architecture is reliably better than A on test synthetic (unseen rules). B equals A (difference -0.004, the 95% CI includes 0).
- All other models are reliably worse than A. D is 0.198 lower than A.
- On counterfactual paired accuracy, all models are reliably lower than A.
- No model is materially above the majority baseline on BoardgameQA, DynaBench, or Super-NaturalInstructions.
- The models do not transfer from the synthetic data to the ProofWriter open-world sets. The gains over A on these sets come with a large seed spread. They are weak evidence.
- A is the fastest model: p50 latency 9.53 ms on an L4 at batch size 1 (lower is better). The D family needs about 2 times this latency without a cache.

## Limitations

- The external baselines (Ettin-32M, DeBERTa-v3-xsmall) are not evaluated. There is no comparison with a model that has different pretraining.
- All 11 models share the Ettin-17M initialization. A contamination affects them equally.
  The sentence frames and names of the ProofWriter depth5 set occur in the Ettin pretraining data.
  Super-NaturalInstructions is not scanned. Treat it as possibly seen in pretraining.
- DynaBench has 543 records, so its CIs are wide.
- Each model has 3 seeds, 1 trial, and 12000 steps. The bootstrap CI covers the record sample only, not the seed variance.
  On the ProofWriter sets, the seed spread is larger than most differences.
- The ProofWriter sets are open-world (True / False / Unknown). Compare them only with published open-world results.
- No mobile or INT8 measurement exists (Step 9 is skipped).

## License

Code, weights, and the synthetic data: Apache-2.0 (`LICENSE`).
`NOTICE` has the attributions of the third-party parts.
`DATA_LICENSES.md` records the license of each data set.
