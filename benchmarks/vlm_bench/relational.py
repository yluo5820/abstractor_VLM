"""Five bounded relational-attention probes for the pinned native LFM2.5 model.

The content stream supplies queries/keys, while a learned symbol stream supplies
values. The native sensory path is preserved. These are not object slots or a
reproduction of Slot Abstractor. Microbatch one, one image, no padded text only.
"""
import math
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F

VARIANTS = ('early', 'late', 'joint', 'memory', 'dual')


def prefix_end(ids, marker):
    """Find exactly one assistant header, never include supervised answer tokens."""
    values = ids.tolist()
    matches = [i + len(marker) for i in range(len(values) - len(marker) + 1)
               if values[i:i + len(marker)] == marker]
    if len(matches) != 1:
        raise ValueError('Expected exactly one assistant header in a single-turn input')
    return matches[0]


def ordered_pool(x, count):
    """Fixed contiguous bins in native token order; deterministic GEMM backward.

    For short text, repeat bins rather than inventing padded content. Bin indices
    supply positions, not object identities or recovered original-image coordinates.
    """
    n = x.shape[-2]
    if n < 1:
        raise ValueError('Empty content stream')
    i = torch.arange(n, device=x.device)
    if n >= count:
        weights = (i[None, :] * count // n == torch.arange(count, device=x.device)[:, None]).float()
        weights = weights / weights.sum(-1, keepdim=True)
    else:
        weights = (i[None, :] == (torch.arange(count, device=x.device) * n // count)[:, None]).float()
    return weights.to(x.dtype) @ x


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.q, self.k, self.v, self.out = [nn.Linear(width, width) for _ in range(4)]

    def forward(self, query, key, value):
        b, n, d = query.shape
        def split(x):
            return x.reshape(b, -1, self.heads, d // self.heads).transpose(1, 2)
        q, k, v = split(self.q(query)), split(self.k(key)), split(self.v(value))
        # Explicit small attention matrices avoid backend-specific fused kernels.
        a = (q @ k.transpose(-1, -2) / math.sqrt(d // self.heads)).softmax(-1)
        return self.out((a @ v).transpose(1, 2).reshape(b, n, d))


class RelationLayer(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.relation = Attention(width, heads)
        self.self_attention = Attention(width, heads)
        self.content_norm = nn.LayerNorm(width)
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(4)])
        self.ff = nn.ModuleList([nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(),
                                              nn.Linear(2 * width, width)) for _ in range(2)])

    def forward(self, content, symbols):
        x = self.content_norm(content)
        symbols = symbols + self.relation(x, x, self.norms[0](symbols))
        symbols = symbols + self.ff[0](self.norms[1](symbols))
        s = self.norms[2](symbols)
        symbols = symbols + self.self_attention(s, s, s)
        return symbols + self.ff[1](self.norms[3](symbols))


class RelationalProbe(nn.Module):
    def __init__(self, variant, vision_dim=1152, text_dim=2048, width=256, heads=4,
                 depth=2, symbols=128):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.spec = dict(variant=variant, vision_dim=vision_dim, text_dim=text_dim,
                         width=width, heads=heads, depth=depth, symbols=symbols)
        self.variant = variant
        input_dim = vision_dim if variant == 'early' else text_dim
        self.content_in = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, width))
        self.symbols = nn.Parameter(torch.empty(1, symbols, width))
        nn.init.normal_(self.symbols, std=0.02)
        self.layers = nn.ModuleList([RelationLayer(width, heads) for _ in range(depth)])
        self.read = Attention(width, heads)
        self.output = nn.Linear(width, input_dim, bias=False)
        # Exactly preserve the base function, without also zeroing the read gate.
        nn.init.zeros_(self.output.weight)
        if variant == 'memory':
            self.query_in = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, width))
            self.gate = nn.Parameter(torch.zeros(()))  # sigmoid(0) = 0.5
        if variant == 'dual':
            self.text_in = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, width))
            self.text_layers = nn.ModuleList([RelationLayer(width, heads) for _ in range(depth)])
            self.fuse = nn.Sequential(nn.LayerNorm(2 * width), nn.Linear(2 * width, width), nn.GELU())

    def abstract(self, x, text=None):
        c = self.content_in(x.float())
        pooled = ordered_pool(c, self.spec['symbols'])
        symbols = self.symbols.expand(c.shape[0], -1, -1)
        for layer in self.layers:
            symbols = layer(pooled, symbols)
        if self.variant == 'dual':
            if text is None:
                raise ValueError('Dual probe requires instruction-only embeddings')
            t = ordered_pool(self.text_in(text.float()), self.spec['symbols'])
            other = self.symbols.expand(c.shape[0], -1, -1)
            for layer in self.text_layers:
                other = layer(t, other)
            symbols = self.fuse(torch.cat([symbols, other], dim=-1))
        return c, pooled, symbols

    def forward(self, x, text=None):
        c, pooled, symbols = self.abstract(x, text)
        return x + self.output(self.read(c, pooled, symbols)).to(x.dtype)

    def read_memory(self, x, memory):
        _, _, symbols = memory
        q = self.query_in(x.float())
        delta = self.gate.sigmoid() * self.output(self.read(q, symbols, symbols))
        return x + delta.to(x.dtype)


class Controller:
    """Hooks keep native image tiling, causal language layers and generation intact."""
    def __init__(self, base, processor, probe):
        self.base, self.probe = base, probe
        self.marker = processor.tokenizer.encode('<|im_start|>assistant\n', add_special_tokens=False)
        self.image_id = base.config.image_token_id
        self.context = None
        self.handles = []
        self.handles.append(base.register_forward_pre_hook(self.before, with_kwargs=True))
        if probe.variant == 'early':
            self.handles.append(base.model.vision_tower.register_forward_hook(self.vision, with_kwargs=True))
        if probe.variant in ('late', 'memory', 'dual'):
            original = base.model.get_image_features
            def features(module, *args, **kwargs):
                result = original(*args, **kwargs)
                tiles = result.pooler_output
                lengths = [len(t) for t in tiles]
                x = torch.cat(tiles, dim=0).unsqueeze(0)
                if self.probe.variant == 'memory':
                    self.context['memory'] = self.probe.abstract(x)
                else:
                    x = self.probe(x, self.context['text'])
                    result.pooler_output = list(x.squeeze(0).split(lengths, dim=0))
                return result
            base.model.get_image_features = MethodType(features, base.model)
        if probe.variant in ('joint', 'memory'):
            if base.config.text_config.layer_types[2] != 'full_attention':
                raise ValueError('Pinned LFM layer 2 is expected to be full attention')
            self.handles.append(base.model.language_model.layers[2].register_forward_hook(self.language))

    def before(self, module, args, kwargs):
        ids = kwargs.get('input_ids', args[0] if args else None)
        if ids is None or ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError('Probe supports exactly one example per forward, with input IDs')
        if kwargs.get('pixel_values') is not None:
            mask = kwargs.get('attention_mask')
            if mask is not None and not bool(mask.all()):
                raise ValueError('Padded language batches are unsupported')
            end = prefix_end(ids[0], self.marker)
            text_ids = ids[:, :end][:, ids[0, :end] != self.image_id]
            self.context = {'end': end, 'prefill': True, 'memory': None,
                            'text': self.base.get_input_embeddings()(text_ids)}
        else:
            if self.context is None or kwargs.get('past_key_values') is None or ids.shape[1] != 1:
                raise ValueError('Text-only call must be a cached one-token continuation of this image')
            self.context['prefill'] = False

    def vision(self, module, args, kwargs, result):
        valid = kwargs.get('pixel_attention_mask')
        if valid is None or valid.shape != result.last_hidden_state.shape[:2]:
            raise ValueError('Missing native vision patch mask')
        valid = valid.bool()
        x = result.last_hidden_state
        changed = self.probe(x[valid].unsqueeze(0)).squeeze(0)
        full = x.clone()
        full[valid] = changed
        result.last_hidden_state = full
        return result

    def language(self, module, args, result):
        if self.probe.variant == 'memory':
            if self.context['memory'] is None:
                raise ValueError('Relational memory was not constructed from the current image')
            return self.probe.read_memory(result, self.context['memory'])
        if not self.context['prefill']:
            return result
        end = self.context['end']
        # Prefix-only noncausal relation attention. Never inspect or modify answer
        # positions; cached decoding sees the same transformed prefix as training.
        return torch.cat([self.probe(result[:, :end]), result[:, end:]], dim=1)


def install(base, processor, variant, state=None):
    if hasattr(base, 'relation_probe'):
        raise ValueError('Probe already installed')
    if base.config.model_type != 'lfm2_vl':
        raise ValueError('Only the audited native LFM2-VL is supported')
    probe = RelationalProbe(variant, vision_dim=base.config.vision_config.hidden_size,
                            text_dim=base.config.text_config.hidden_size)
    probe.to(device=next(base.parameters()).device, dtype=torch.float32)
    if state is not None:
        probe.load_state_dict(state, strict=True)
    base.add_module('relation_probe', probe)
    controller = Controller(base, processor, probe)
    base._relational_controller = controller
    return controller
