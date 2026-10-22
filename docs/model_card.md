---
license: apache-2.0
base_model: jhu-clsp/ettin-encoder-17m
pipeline_tag: text-classification
datasets:
- 0xanto/crux-data
language:
- en
tags:
- pytorch
- safetensors
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

## Usage

The models are not `transformers` models. They need the code of the project.
Clone https://github.com/0xAnto/crux, run `pip install -r requirements.txt`, and run this script in the clone:

```python
# Run in a clone of https://github.com/0xAnto/crux after `pip install -r requirements.txt`.
import torch
import yaml
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

from crux.data.batching import make_dataloader
from crux.models import build_model
from crux.train import input_mode, predict

name, seed = "A", "s0"  # 11 models: A, A-param, A+CF, B, B-abl, B+CF, C, C+cache, C+equivariant, D, D-CF; seeds s0, s1, s2
cfg = yaml.safe_load(open(hf_hub_download("0xanto/crux", f"weights/{name}/{seed}/config.yaml")))
model = build_model(cfg["model"])
model.load_state_dict(load_file(hf_hub_download("0xanto/crux", f"weights/{name}/{seed}/model.safetensors")))
dcfg = yaml.safe_load(open(cfg["data_config"]))  # configs/step2_data.yaml of the clone
tokenizer = AutoTokenizer.from_pretrained(dcfg["tokenizer"])

record = {
    "id": "example-1", "source": "example", "task_type": "binary", "meta": {}, "target": 0,
    "state": "Anne is kind. Anne is young.",
    "instruction": "§1 If a person is kind and young, then this person is happy. Question: Is it established that Anne is happy?",
    "options": ["True", "False", "Unknown"],
}
loader = make_dataloader([record], tokenizer, dcfg, input_mode(cfg["model"]), joint_options=True)
probs = predict(model, loader, torch.device("cpu"), fp16=False)["example-1"]
print({o: round(float(p), 3) for o, p in zip(record["options"], probs)})
```

A record has a `state` (facts), an `instruction` (rules, then the question after `Question:`), and `options`.
The output is one probability for each option.
The output of the script for seed s0 is: A `{'True': 0.78, 'False': 0.218, 'Unknown': 0.002}`, B `{'True': 0.926, 'False': 0.024, 'Unknown': 0.05}`, D `{'True': 0.28, 'False': 0.258, 'Unknown': 0.462}`.
To evaluate a model on the data, use `python -m crux.benchmark` (see the code repository).

## Training

- Base model: `jhu-clsp/ettin-encoder-17m` (7 layers, hidden size 256, 4 attention heads).
- Data: the synthetic train split of https://huggingface.co/datasets/0xanto/crux-data (75,646 records).
- The same budget for all models: 12000 steps of 32 records, 1 trial (learning rate 3e-5, weight decay 0.01 on the weight matrices), AdamW, 6% warm-up, gradient clip 1.0, fp16 mixed precision.
- Seeds: 13, 42, and 2026 (folders `s0`, `s1`, `s2`).
- The checkpoint with the best validation `unseen_rule_accuracy` (evaluation every 1000 steps) is the released checkpoint.
- Hardware: one NVIDIA L4 (Google Colab) for each run.

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

## Citation

```bibtex
@misc{ranjith2026crux,
  author       = {Antony Ranjith F.},
  title        = {Does Rule Conditioning Help a 17M Encoder Decide on Unseen Rules? A Negative Result for the {CRUX} Architectures},
  year         = {2026},
  howpublished = {\url{https://github.com/0xAnto/crux}}
}
```
