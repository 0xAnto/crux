"""Step 5: the models A, B (Rule Compiler), B-ablation, and the A-param control.

All models score each option sequence [CLS] state [SEP] instruction [SEP] option [SEP] with one scalar.
The softmax over the real options of a record gives the probabilities. Padded options are not encoded.

- A: Ettin encoder, mean pooling, decision head (dense, GELU, LayerNorm, linear to 1). Full fine-tune.
- B: A + a hypernetwork. The instruction embedding z is the mean hidden state over the instruction tokens
  (between the first and the second [SEP]), averaged over the options of a record. The hypernetwork
  generates rank-r factors (A, B) for the head dense layer: W x + (x A) B. With hyper_layers K > 0 it also
  generates rank-r adapters for Wqkv and Wo of the top K encoder layers. z then comes from the hidden states
  after layer L-K-1, so one encoder pass is enough. The B factors start at zero, so B starts equal to A.
  The encoder factors are bounded (see `bound`): no norm follows the delta, and unbounded factors overflow fp16.
- A-param: A + a static LoRA with the same number of trainable parameters as the hypernetwork of B.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from transformers.masking_utils import create_bidirectional_mask, create_bidirectional_sliding_window_mask

ARCHS = ("A", "B", "A-param")


class Lora(nn.Module):
    """A linear layer plus a low-rank delta x -> base(x) + (x A) B.

    The delta is the pair `delta` that the hypernetwork sets before a pass (one pair for each row of x),
    else the static parameters A, B (after `add_static`), else nothing.
    """

    def __init__(self, base):
        super().__init__()
        self.base, self.delta, self.A, self.B = base, None, None, None
        self.in_features, self.out_features = base.in_features, base.out_features

    def add_static(self, rank):
        self.A = nn.Parameter(torch.empty(self.in_features, rank))
        nn.init.uniform_(self.A, -self.in_features ** -0.5, self.in_features ** -0.5)  # A random, B zero
        self.B = nn.Parameter(torch.zeros(rank, self.out_features))

    def forward(self, x):
        y = self.base(x)
        a, b = self.delta if self.delta is not None else (self.A, self.B)
        return y if a is None else y + (x @ a) @ b


class Hypernet(nn.Module):
    """z [E, d] -> one pair (A [E, in, r], B [E, r, out]) for each (in, out) in `dims`.

    LayerNorm, Linear(d, hidden), GELU, and one Linear to all factors. The B factors have zero weights and
    zero bias, so every delta is zero at the start. The A factors start small and random, so the B factors
    get a gradient from the first step.
    """

    def __init__(self, d, dims, rank, hidden):
        super().__init__()
        self.dims, self.rank = dims, rank
        self.norm, self.fc1 = nn.LayerNorm(d), nn.Linear(d, hidden)
        self.fc2 = nn.Linear(hidden, sum(rank * (i + o) for i, o in dims))
        with torch.no_grad():
            self.fc2.weight.normal_(std=0.02)
            self.fc2.bias.zero_()
            pos = 0
            for i, o in self.dims:
                self.fc2.bias[pos:pos + rank * i].uniform_(-i ** -0.5, i ** -0.5)
                self.fc2.weight[pos + rank * i:pos + rank * (i + o)] = 0  # the B factor
                self.fc2.bias[pos + rank * i:pos + rank * (i + o)] = 0
                pos += rank * (i + o)

    def forward(self, z):
        out, pairs, pos, r = self.fc2(F.gelu(self.fc1(self.norm(z)))), [], 0, self.rank
        for i, o in self.dims:
            a = out[:, pos:pos + r * i].view(-1, i, r)
            b = out[:, pos + r * i:pos + r * (i + o)].view(-1, r, o)
            pairs.append((a, b))
            pos += r * (i + o)
        return pairs


def bound(v, limit):
    """Soft bound of the generated factors of the encoder adapters: limit * tanh(v / limit), zero stays zero."""
    return limit * torch.tanh(v / limit)


def static_ranks(costs, budget):
    """Ranks for modules with `costs` (in + out of each module) so that sum(rank * cost) is closest to
    `budget` and not above it. Start with one rank for all, then add one rank at a time, from the first module."""
    r = budget // sum(costs)
    ranks, left = [r] * len(costs), budget - r * sum(costs)
    while any(c <= left for c in costs):
        for k, c in enumerate(costs):
            if c <= left:
                ranks[k] += 1
                left -= c
    return ranks


def span_mean(h, ids, mask, sep_id):
    """Mean of h [N, L, d] over the tokens between the first and the second [SEP] of each row."""
    sep = ids == sep_id
    m = ((sep.long().cumsum(1) == 1) & ~sep & mask.bool()).unsqueeze(-1).float()
    return (h.float() * m).sum(1) / m.sum(1).clamp(min=1)


class Scorer(nn.Module):
    def __init__(self, encoder, arch="A", rank=8, hyper_layers=0, hyper_hidden=64):
        """`hyper_layers` K: the top K encoder layers get generated adapters (B only). For A-param it names
        the B variant whose trainable parameter count the static LoRA matches."""
        super().__init__()
        if arch not in ARCHS or (arch == "A" and hyper_layers):
            raise ValueError(f"unknown arch {arch!r}, or hyper_layers with arch A")
        c = encoder.config
        self.encoder, self.arch, self.sep_id = encoder, arch, c.sep_token_id
        self.K = hyper_layers if arch == "B" else 0
        d, layers = c.hidden_size, encoder.layers
        self.dense = Lora(nn.Linear(d, d, bias=False))
        self.norm, self.out = nn.LayerNorm(d), nn.Linear(d, 1)
        # B: the adapted linears are Wqkv and Wo of the top K layers.
        top = [(l.attn, n) for l in layers[len(layers) - hyper_layers:] for n in ("Wqkv", "Wo")]
        dims = [(d, d)] + [(getattr(p, n).in_features, getattr(p, n).out_features) for p, n in top]
        if arch == "B":
            self.hyper = Hypernet(d, dims, rank, hyper_hidden)
            self.adapted = [self._wrap(p, n) for p, n in top]
        elif arch == "A-param":
            budget = sum(p.numel() for p in Hypernet(d, dims, rank, hyper_hidden).parameters())
            mods = [(self, "dense")] + [(m, n) for l in reversed(layers)
                                        for m, n in ((l.attn, "Wqkv"), (l.attn, "Wo"), (l.mlp, "Wi"), (l.mlp, "Wo"))]
            costs = [getattr(m, n).in_features + getattr(m, n).out_features for m, n in mods]
            for (m, n), r in zip(mods, static_ranks(costs, budget)):
                if r:
                    self._wrap(m, n).add_static(r)
            self.static_target = budget

    @staticmethod
    def _wrap(parent, name):
        mod = getattr(parent, name)
        if not isinstance(mod, Lora):
            mod = Lora(mod)
            setattr(parent, name, mod)
        return mod

    def extra_params(self):
        """Trainable parameters that A does not have: the hypernetwork (B) or the static LoRA (A-param)."""
        if self.arch == "B":
            return sum(p.numel() for p in self.hyper.parameters())
        return sum(m.A.numel() + m.B.numel() for m in self.modules() if isinstance(m, Lora) and m.A is not None)

    def pretrained_params(self):
        """The pretrained encoder parameters (the other parameters are new)."""
        new = {id(p) for m in self.modules() if isinstance(m, Lora) for p in (m.A, m.B) if p is not None}
        return [p for p in self.encoder.parameters() if id(p) not in new]

    def forward(self, input_ids, attention_mask, option_mask, **_):
        """input_ids, attention_mask [B, K, L], option_mask [B, K] -> logits [B, K] (-inf for padded options)."""
        enc, n_ex = self.encoder, option_mask.shape[0]
        ex = option_mask.nonzero()[:, 0]  # the record of each real option sequence
        ids, mask = input_ids[option_mask], attention_mask[option_mask]
        h = enc.embeddings(input_ids=ids)
        kw = dict(config=enc.config, inputs_embeds=h, attention_mask=mask)
        masks = {"full_attention": create_bidirectional_mask(**kw),
                 "sliding_attention": create_bidirectional_sliding_window_mask(**kw)}
        pos_ids = torch.arange(ids.shape[1], device=ids.device)[None]
        pos = {t: enc.rotary_emb(h, pos_ids, t) for t in set(enc.config.layer_types)}

        def run(layers, h):
            for l in layers:
                h = l(h, attention_mask=masks[l.attention_type], position_embeddings=pos[l.attention_type])
            return h

        def instruction_embedding(h):  # one z for each record: the mean over its options
            z = torch.zeros(n_ex, h.shape[-1], device=h.device).index_add_(0, ex, span_mean(h, ids, mask, self.sep_id))
            return z / torch.bincount(ex, minlength=n_ex).clamp(min=1)[:, None]

        low = len(enc.layers) - self.K
        h = run(enc.layers[:low], h)
        pairs = None
        if self.arch == "B" and self.K:
            pairs = self.hyper(instruction_embedding(h))
            for m, (a, b) in zip(self.adapted, pairs[1:]):
                m.delta = (bound(a, m.in_features ** -0.5)[ex], bound(b, m.out_features ** -0.5)[ex])
        h = enc.final_norm(run(enc.layers[low:], h))
        for m in getattr(self, "adapted", []):
            m.delta = None
        if self.arch == "B" and not self.K:
            pairs = self.hyper(instruction_embedding(h))
        self.dense.delta = None if pairs is None else (pairs[0][0][ex], pairs[0][1][ex])
        m = mask.unsqueeze(-1).float()
        pooled = ((h.float() * m).sum(1) / m.sum(1)).unsqueeze(1)  # mean pooling, [N, 1, d]
        score = self.out(self.norm(F.gelu(self.dense(pooled)))).view(-1).float()
        self.dense.delta = None
        logits = torch.full(option_mask.shape, float("-inf"), device=score.device)
        logits[option_mask] = score
        return logits


def build_model(cfg, encoder=None):
    """cfg: the `model` section of the config. `encoder` is for tests (a tiny random ModernBERT)."""
    if encoder is None:
        encoder = AutoModel.from_pretrained(cfg["id"], attn_implementation=cfg.get("attn_implementation", "sdpa"))
    return Scorer(encoder, cfg["arch"], cfg.get("rank", 8), cfg.get("hyper_layers", 0), cfg.get("hyper_hidden", 64))


def count_params(model):
    return {"params_total": sum(p.numel() for p in model.parameters()),
            "params_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "params_extra": model.extra_params()}
