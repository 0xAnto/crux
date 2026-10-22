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
- External baselines (Step 8): A on an encoder that is not ModernBERT (DeBERTa-v3-xsmall). The encoder runs on the option
  sequences with its own `forward`; the pooling and the head are those of A. Only arch A and the joint encoding use this path.

Step 7 (architecture D = C + two options of the model, both off by default):
- `joint_layers` J > 0 (split encoding, as in DeFormer): the input is [CLS] state [SEP], [CLS] instruction [SEP], and
  [CLS] option [SEP] as three separate sequences. The bottom L-J encoder layers encode each one alone, with positions
  from 0. The state and the instruction states of these layers do not depend on the options or on each other: they
  are the cacheable part (`encode_bottom`). The top J layers run on [state ; instruction ; option] for each option.
  The real tokens are moved to the front of the sequence, so the positions and the sliding window do not depend on
  the padding. J = 0 is the joint encoding of A, B, and C.
- `option_attention`: after the pooling, one set-attention block mixes the pooled vectors of the options of a
  record. It has no position information, so the scores are permutation-equivariant.
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


def record_mean(x, ex, n_ex):
    """Mean of the rows x [N, d] over the options of each record: ex [N] is the record of each row -> [n_ex, d]."""
    z = torch.zeros(n_ex, x.shape[-1], device=x.device).index_add_(0, ex, x)
    return z / torch.bincount(ex, minlength=n_ex).clamp(min=1)[:, None]


def span_mean(h, ids, mask, sep_id):
    """Mean of h [N, L, d] over the tokens between the first and the second [SEP] of each row."""
    sep = ids == sep_id
    m = ((sep.long().cumsum(1) == 1) & ~sep & mask.bool()).unsqueeze(-1).float()
    return (h.float() * m).sum(1) / m.sum(1).clamp(min=1)


class OptionAttention(nn.Module):
    """One set-attention block over the options of a record: x + out(attention(LayerNorm(x))).

    No position information, and a key padding mask for the padded options. Thus a permutation of the options
    permutes the output. The output projection starts at zero, so the block is the identity at the start
    (a LayerNorm after the residual would break this).
    """

    def __init__(self, d, heads):
        super().__init__()
        self.norm, self.attn = nn.LayerNorm(d), nn.MultiheadAttention(d, heads, batch_first=True)
        with torch.no_grad():
            self.attn.out_proj.weight.zero_()
            self.attn.out_proj.bias.zero_()

    def forward(self, x, option_mask):
        """x [N, d]: the pooled vectors of the real options, in the row order of option_mask [B, K]."""
        full = x.new_zeros(*option_mask.shape, x.shape[-1])
        full[option_mask] = self.norm(x)
        a = self.attn(full, full, full, key_padding_mask=~option_mask, need_weights=False)[0]
        return x + a[option_mask]


class Scorer(nn.Module):
    def __init__(self, encoder, arch="A", rank=8, hyper_layers=0, hyper_hidden=64, joint_layers=0, option_attention=False):
        """`hyper_layers` K: the top K encoder layers get generated adapters (B only). For A-param it names
        the B variant whose trainable parameter count the static LoRA matches.
        `joint_layers` J: 0 is the joint encoding. J > 0 is the split encoding with J joint top layers (Step 7).
        `option_attention`: add the set-attention block over the options (Step 7)."""
        super().__init__()
        if arch not in ARCHS or (arch == "A" and hyper_layers):
            raise ValueError(f"unknown arch {arch!r}, or hyper_layers with arch A")
        c = encoder.config
        self.generic = c.model_type != "modernbert"  # an encoder with its own forward (Step 8): arch A only
        if self.generic and (arch != "A" or joint_layers):
            raise ValueError("an encoder that is not ModernBERT supports only arch A with joint_layers 0")
        if joint_layers and (hyper_layers or not 0 < joint_layers <= c.num_hidden_layers):
            raise ValueError("joint_layers must be in 1..L, and it does not combine with hyper_layers")
        self.encoder, self.arch, self.sep_id, self.J = encoder, arch, getattr(c, "sep_token_id", None), joint_layers
        self.K = hyper_layers if arch == "B" else 0
        d, layers = c.hidden_size, [] if self.generic else encoder.layers
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
        self.option_attn = OptionAttention(d, c.num_attention_heads) if option_attention else None  # last: no RNG use if off

    @staticmethod
    def _wrap(parent, name):
        mod = getattr(parent, name)
        if not isinstance(mod, Lora):
            mod = Lora(mod)
            setattr(parent, name, mod)
        return mod

    def extra_params(self):
        """Trainable parameters that A does not have: the hypernetwork (B) or the static LoRA (A-param), and the
        set-attention block. The split encoding (joint_layers) adds no parameter."""
        if self.arch == "B":
            n = sum(p.numel() for p in self.hyper.parameters())
        else:
            n = sum(m.A.numel() + m.B.numel() for m in self.modules() if isinstance(m, Lora) and m.A is not None)
        return n + (sum(p.numel() for p in self.option_attn.parameters()) if self.option_attn is not None else 0)

    def pretrained_params(self):
        """The pretrained encoder parameters (the other parameters are new)."""
        new = {id(p) for m in self.modules() if isinstance(m, Lora) for p in (m.A, m.B) if p is not None}
        return [p for p in self.encoder.parameters() if id(p) not in new]

    def _runner(self, h, mask):
        """Return run(layers, h) for the sequences h [N, L, d] with attention mask [N, L]. Positions are 0..L-1."""
        enc = self.encoder
        kw = dict(config=enc.config, inputs_embeds=h, attention_mask=mask)
        masks = {"full_attention": create_bidirectional_mask(**kw),
                 "sliding_attention": create_bidirectional_sliding_window_mask(**kw)}
        pos_ids = torch.arange(h.shape[1], device=h.device)[None]
        pos = {t: enc.rotary_emb(h, pos_ids, t) for t in set(enc.config.layer_types)}

        def run(layers, h):
            for l in layers:
                h = l(h, attention_mask=masks[l.attention_type], position_embeddings=pos[l.attention_type])
            return h
        return run

    def encode_bottom(self, ids, mask):
        """The cacheable part of the split encoding (joint_layers > 0). ids, mask [N, L] are separate sequences
        [CLS] text [SEP]. Return their hidden states [N, L, d] after the bottom L-J layers. A state or an
        instruction gives the same result in every batch, so a caller can keep it and pass it to `forward`."""
        h = self.encoder.embeddings(input_ids=ids)
        return self._runner(h, mask)(self.encoder.layers[:len(self.encoder.layers) - self.J], h)

    def _split(self, option_mask, ex, state_mask, instruction_mask, option_ids, option_attention_mask,
               state_states, instruction_states):
        """Split encoding: the top J layers on [state ; instruction ; option] for each real option.
        Return the final hidden states [N, L', d], their mask, and the generated factors (B) or None."""
        enc, n_ex = self.encoder, option_mask.shape[0]
        om = option_attention_mask[option_mask]
        ho = self.encode_bottom(option_ids[option_mask], om)
        # The instruction tokens without [CLS] and [SEP] (the span of z, as in the joint encoding).
        idx, n = torch.arange(instruction_mask.shape[1], device=ho.device)[None], instruction_mask.sum(1, keepdim=True)
        span = (idx > 0) & (idx < n - 1) & instruction_mask.bool()
        h = torch.cat([state_states[ex], instruction_states[ex], ho], 1)
        mask = torch.cat([state_mask[ex], instruction_mask[ex], om], 1).bool()
        span = torch.cat([torch.zeros_like(state_mask[ex]).bool(), span[ex], torch.zeros_like(om).bool()], 1)
        # Move the real tokens to the front: the pads of the segments leave the sequence.
        order = torch.sort((~mask).to(torch.uint8), dim=1, stable=True).indices[:, :int(mask.sum(1).max())]
        h = h.gather(1, order[..., None].expand(-1, -1, h.shape[-1]))
        mask, span = mask.gather(1, order), span.gather(1, order)[..., None].float()
        mask = mask.long()
        h = enc.final_norm(self._runner(h, mask)(enc.layers[len(enc.layers) - self.J:], h))
        pairs = None
        if self.arch == "B":
            pairs = self.hyper(record_mean((h.float() * span).sum(1) / span.sum(1).clamp(min=1), ex, n_ex))
        return h, mask, pairs

    def _score(self, h, mask, pairs, option_mask, ex):
        """h [N, L, d] final hidden states, mask [N, L] -> logits [B, K] (-inf for padded options)."""
        self.dense.delta = None if pairs is None else (pairs[0][0][ex], pairs[0][1][ex])
        m = mask.unsqueeze(-1).float()
        pooled = ((h.float() * m).sum(1) / m.sum(1)).unsqueeze(1)  # mean pooling, [N, 1, d]
        if self.option_attn is not None:
            pooled = self.option_attn(pooled[:, 0], option_mask).unsqueeze(1)
        score = self.out(self.norm(F.gelu(self.dense(pooled)))).view(-1).float()
        self.dense.delta = None
        logits = torch.full(option_mask.shape, float("-inf"), device=score.device)
        logits[option_mask] = score
        return logits

    def forward(self, input_ids=None, attention_mask=None, option_mask=None, state_ids=None, state_mask=None,
                instruction_ids=None, instruction_mask=None, option_ids=None, option_attention_mask=None,
                state_cache=None, instruction_cache=None, **_):
        """Joint encoding (joint_layers 0): input_ids, attention_mask [B, K, L], option_mask [B, K].
        Split encoding: state_ids, state_mask [B, Ls], instruction_ids, instruction_mask [B, Li], option_ids,
        option_attention_mask [B, K, Lo], option_mask [B, K]. `state_cache` and `instruction_cache` [B, L, d]
        (from `encode_bottom`) replace the bottom layers of that segment, and its ids are not read.
        Return logits [B, K] (-inf for padded options)."""
        enc, n_ex = self.encoder, option_mask.shape[0]
        ex = option_mask.nonzero()[:, 0]  # the record of each real option sequence
        if self.J:
            h, mask, pairs = self._split(
                option_mask, ex, state_mask, instruction_mask, option_ids, option_attention_mask,
                self.encode_bottom(state_ids, state_mask) if state_cache is None else state_cache,
                self.encode_bottom(instruction_ids, instruction_mask) if instruction_cache is None else instruction_cache)
            return self._score(h, mask, pairs, option_mask, ex)
        ids, mask = input_ids[option_mask], attention_mask[option_mask]
        if self.generic:
            return self._score(enc(input_ids=ids, attention_mask=mask).last_hidden_state, mask, None, option_mask, ex)
        h = enc.embeddings(input_ids=ids)
        run = self._runner(h, mask)

        def instruction_embedding(h):  # one z for each record: the mean over its options
            return record_mean(span_mean(h, ids, mask, self.sep_id), ex, n_ex)

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
        return self._score(h, mask, pairs, option_mask, ex)


def build_model(cfg, encoder=None):
    """cfg: the `model` section of the config. `encoder` is for tests (a tiny random encoder)."""
    if encoder is None:
        # dtype fp32: the DeBERTa checkpoint is fp16, and transformers 5 keeps the checkpoint dtype. The Ettin checkpoints are fp32.
        encoder = AutoModel.from_pretrained(cfg["id"], attn_implementation=cfg.get("attn_implementation", "sdpa"),
                                            dtype=torch.float32)
    return Scorer(encoder, cfg["arch"], cfg.get("rank", 8), cfg.get("hyper_layers", 0), cfg.get("hyper_hidden", 64),
                  cfg.get("joint_layers", 0), cfg.get("option_attention", False))


def count_params(model):
    return {"params_total": sum(p.numel() for p in model.parameters()),
            "params_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "params_extra": model.extra_params()}
