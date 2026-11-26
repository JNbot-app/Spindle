"""The Spindle network in JAX/Flax, used for fine-tuning.

A Simple Attention Network with:
  * GQA attention with causal convolution taps on Q, K and V, and a sigmoid output gate,
  * a Hadamard MLP instead of a dense FFN. Its mixing factors are Monarch (one matrix
    per group) or Kronecker (one matrix shared by every group),
  * engram n-gram memory, read by a gather and added at a few layers,
  * multi-lane hyper-connections (mHC) between the blocks.

Parameter names match the .cact archive loader in `spindle.archive`.
"""
import math
from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.nn.initializers as jinit
import jax.numpy as jnp
import numpy as np

from . import quantize


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    out_vocab: int = 0
    d_model: int = 768
    num_heads: int = 12
    num_kv_heads: int = 2
    num_layers: int = 20
    qk_head_dim: int = 48
    v_head_dim: int = 64
    max_seq_len: int = 8192
    pad_token_id: int = 0
    rope_theta: float = 100000.0
    sliding_window: int = 0
    global_layers: tuple = ()
    qkv_conv_taps: int = 3
    mhc_lanes: int = 4
    engram_orders: tuple = (2, 3)
    engram_heads: int = 3
    engram_slots: int = 18432
    engram_sub_dim: int = 128
    engram_seed_heads: int = 0
    engram_layers: tuple = (3, 7, 11, 15, 19)
    monarch: bool = True
    kv_window: int = 0
    kv_bits: int = 8
    skip_layers: tuple = ()        # layers that answer tokens skip (decode skip, see _Layer)
    remat: bool = True


def _init():
    return jinit.normal(stddev=0.02)


def _identity_taps(key, shape):
    return jnp.zeros(shape).at[0].set(1.0)


class ZCRMSNorm(nn.Module):
    """Zero-centred RMS norm: the scale is stored as (gain - 1)."""
    epsilon: float = 1e-6

    @nn.compact
    def __call__(self, x):
        scale = self.param('scale', jinit.zeros, (x.shape[-1],))
        rms = jnp.sqrt(jnp.mean(x.astype(jnp.float32) ** 2, axis=-1, keepdims=True) + self.epsilon)
        return ((1 + scale) * x / rms).astype(jnp.float32)


def rope_tables(head_dim, length, theta):
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    angles = jnp.outer(jnp.arange(length, dtype=jnp.float32), freqs)
    return jnp.cos(angles), jnp.sin(angles)


def apply_rope(x, cos, sin):
    T, half = x.shape[2], x.shape[-1] // 2
    cos, sin = cos[:T][None, None], sin[:T][None, None]
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def _shift_right(x, offset):
    if offset == 0:
        return x
    pad = [(0, 0)] * x.ndim
    pad[1] = (offset, 0)
    return jnp.pad(x, pad)[:, :x.shape[1]]


def _mask_diagonal(mask, offset):
    """For each position t: may it see position t - offset? Shape [B, T]."""
    m = mask[:, 0]
    T = m.shape[-1]
    if offset >= T:
        return jnp.zeros(m.shape[:-2] + (T,), m.dtype)
    d = jnp.diagonal(m, offset=-offset, axis1=-2, axis2=-1)
    return d if offset == 0 else jnp.pad(d, ((0, 0), (offset, 0)))


def _rms_unit(x, epsilon=1e-6):
    x = x.astype(jnp.float32)
    return x * jax.lax.rsqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + epsilon)


def _sinkhorn(logits, iters=20):
    def normalize(_, log_k):
        log_k = log_k - jax.nn.logsumexp(log_k, axis=-1, keepdims=True)
        return log_k - jax.nn.logsumexp(log_k, axis=-2, keepdims=True)
    return jnp.exp(jax.lax.fori_loop(0, iters, normalize, logits))


# ---------------------------------------------------------------- engram memory

ENGRAM_CONV_TAPS = 4
_ENGRAM_SEED = 0x9E3779B9
_ENGRAM_PRIME = 0x01000193


def engram_indices(tokens, orders, heads, slots, seed_heads=0):
    """Hash every n-gram ending at each position into `slots` rows, once per (order, head)."""
    u = tokens.astype(jnp.uint32)
    stride = seed_heads or heads
    idx = []
    for oi, order in enumerate(orders):
        for h in range(heads):
            acc = jnp.full_like(u, jnp.uint32((_ENGRAM_SEED * (oi * stride + h + 1)) & 0xFFFFFFFF))
            for j in range(order):
                acc = (acc ^ _shift_right(u, j)) * jnp.uint32(_ENGRAM_PRIME)
            acc = acc ^ (acc >> jnp.uint32(15))
            idx.append((acc % jnp.uint32(slots)).astype(jnp.int32))
    return jnp.stack(idx, axis=-1)


class Engram(nn.Module):
    d_model: int
    num_tables: int
    slots: int
    sub_dim: int
    conv_dilation: int

    @nn.compact
    def __call__(self, indices, ngram_ok, tap_ok):
        tables = self.param('embedding', _init(), (self.num_tables, self.slots, self.sub_dim))
        fetched = tables[jnp.arange(self.num_tables), indices] * ngram_ok[..., None]
        e = quantize.a8(fetched.reshape(*indices.shape[:2], -1))
        k = nn.Dense(self.d_model, use_bias=False, name='key_proj')(e)
        v = nn.Dense(self.d_model, use_bias=False, name='value_proj')(e)
        taps = self.param('taps', _identity_taps, (ENGRAM_CONV_TAPS, self.d_model))
        v = sum(taps[j] * _shift_right(v, j * self.conv_dilation) * tap_ok[j][..., None]
                for j in range(ENGRAM_CONV_TAPS))
        return k, v


# ---------------------------------------------------------------- attention

class Attention(nn.Module):
    """GQA attention. `span` (training only) sows the span attention loss.
    `keep` [B, T] (decode skip): tokens that skip this layer give zero q/k/v to the conv taps of later tokens."""
    config: ModelConfig

    @nn.compact
    def __call__(self, x, mask, rope, layer, span=None, keep=None):
        c = self.config
        B = x.shape[0]
        H, KV, qd, vd = c.num_heads, c.num_kv_heads, c.qk_head_dim, c.v_head_dim
        x = quantize.a8(x)
        q = nn.Dense(H * qd, use_bias=False, name='q_proj')(x)
        k = nn.Dense(KV * qd, use_bias=False, name='k_proj')(x)
        v = nn.Dense(KV * vd, use_bias=False, name='v_proj')(x)

        if c.qkv_conv_taps:
            n = c.qkv_conv_taps
            taps = {name: self.param(f'{name}_taps', _identity_taps, (n, z.shape[-1]))
                    for name, z in (('q', q), ('k', k), ('v', v))}
            ok = [None] + [_mask_diagonal(mask, j)[..., None] for j in range(1, n)]

            def tapped(z, j):
                if j and keep is not None:
                    z = z * keep[..., None]
                shifted = _shift_right(z, j)
                return shifted if ok[j] is None else shifted * ok[j]
            q = sum(taps['q'][j] * tapped(q, j) for j in range(n))
            k = sum(taps['k'][j] * tapped(k, j) for j in range(n))
            v = sum(taps['v'][j] * tapped(v, j) for j in range(n))

        q = q.reshape(B, -1, H, qd).transpose(0, 2, 1, 3)
        k = k.reshape(B, -1, KV, qd).transpose(0, 2, 1, 3)
        v = v.reshape(B, -1, KV, vd).transpose(0, 2, 1, 3)
        q = apply_rope(ZCRMSNorm(name='q_norm')(q), *rope)
        k = apply_rope(ZCRMSNorm(name='k_norm')(k), *rope)
        q, k, v = (quantize.a8_per_head(z) for z in (q, k, v))

        if span is not None:
            from spindle.training.spans import span_loss
            self.sow('intermediates', 'span_loss', span_loss(q, k, mask, *span, layer))

        k = jnp.repeat(k, H // KV, axis=1)
        v = jnp.repeat(v, H // KV, axis=1)
        scores = (q @ k.transpose(0, 1, 3, 2)) / jnp.sqrt(jnp.float32(qd))
        scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)
        out = (nn.softmax(scores, axis=-1) @ v).transpose(0, 2, 1, 3).reshape(B, -1, H * vd)
        out = out * nn.sigmoid(nn.Dense(H * vd, use_bias=False, name='gate_proj')(x))
        return nn.Dense(c.d_model, use_bias=False, name='out_proj')(quantize.a8(out))


# ---------------------------------------------------------------- Hadamard MLP

HADAMARD_COND_RANK = 8
_HADAMARD_PERM_SEEDS = (11, 13)


def _walsh(n):
    h = np.array([[1.0]], np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


def hadamard_blocks(n):
    """Split n into two group sizes (na, nb) with na * nb = n."""
    b = 1 << ((n - 1).bit_length() // 2)
    return b, n // b


def hadamard_perms(n):
    return tuple(np.random.RandomState(s).permutation(n) for s in _HADAMARD_PERM_SEEDS)


def kron_apply(z, a, b):
    """z viewed as [na, nb]; mix rows with A and columns with B (one shared matrix each)."""
    lead = z.shape[:-1]
    z = jnp.einsum('...ij,ik,jl->...kl', z.reshape(*lead, a.shape[0], b.shape[0]), a, b)
    return z.reshape(*lead, -1)


def monarch_apply(z, a, b):
    """Like kron_apply, but every group has its own matrix (same compute).

    z is viewed as [na, nb]. b [na, nb, nb] mixes each row i with its own b[i];
    then a [nb, na, na] mixes each column l with its own a[l]. With every b[i] equal
    to B and every a[l] equal to A this is exactly kron_apply(z, A, B).
    """
    lead = z.shape[:-1]
    z = z.reshape(*lead, b.shape[0], b.shape[1])
    z = jnp.einsum('...ij,ijl->...il', z, b)
    z = jnp.einsum('...il,lik->...kl', z, a)
    return z.reshape(*lead, -1)


class HadamardMLP(nn.Module):
    """Three structured mixing stages with diagonal scales, a SiLU and a low-rank condition."""
    d_model: int
    monarch: bool = True

    @nn.compact
    def __call__(self, x):
        n = 1 << (self.d_model - 1).bit_length()
        na, nb = hadamard_blocks(n)
        p1, p2 = (jnp.asarray(p) for p in hadamard_perms(n))
        if self.monarch:
            mix = monarch_apply
            shapes = ((nb, na, na), (na, nb, nb))
            inits = (np.repeat(_walsh(na)[None], nb, 0), np.repeat(_walsh(nb)[None], na, 0))
        else:
            mix = kron_apply
            shapes = ((na, na), (nb, nb))
            inits = (_walsh(na), _walsh(nb))
        factors = [tuple(self.param(f'w{s}{side}', lambda *_, w=w: jnp.asarray(w), shape)
                         for side, shape, w in zip('ab', shapes, inits)) for s in (1, 2, 3)]
        d1, d2, d3, d4 = (self.param(f'd{i}', jinit.ones, (n,)) for i in (1, 2, 3, 4))
        b2 = self.param('b2', jinit.zeros, (n,))
        cond_v = self.param('cond_v', _init(), (self.d_model, HADAMARD_COND_RANK))
        cond_u = self.param('cond_u', jinit.zeros, (HADAMARD_COND_RANK, n))
        cond = 1 + nn.softmax(x @ cond_v, axis=-1) @ cond_u
        z = jnp.pad(x, ((0, 0), (0, 0), (0, n - self.d_model))) if n > self.d_model else x
        z = mix(d1 * z, *factors[0])[..., p1]
        z = mix(nn.silu(d2 * cond * z + b2), *factors[1])[..., p2]
        z = mix(d3 * z, *factors[2])
        return (d4 * z)[..., :self.d_model]


# ---------------------------------------------------------------- blocks and stack

class Block(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, x, mask, rope, engram_kv, site_flags, layer, span, keep=None):
        if engram_kv is not None:
            ek, ev = engram_kv
            alpha = nn.sigmoid(jnp.einsum('btd,sbtd->sbt', _rms_unit(x), _rms_unit(ek))
                               / math.sqrt(self.config.d_model))
            x = x + jnp.einsum('s,sbt,sbtd->btd', site_flags, alpha, ev.astype(jnp.float32))
        skip = x
        x = Attention(self.config, name='self_attn')(ZCRMSNorm()(x), mask, rope, layer, span, keep)
        gate = nn.sigmoid(self.param('attn_gate', jinit.zeros, ()))
        x = skip + gate * ZCRMSNorm(name='post_attn_norm')(x)
        skip = x
        x = HadamardMLP(self.config.d_model, self.config.monarch, name='hadamard_mlp')(
            ZCRMSNorm(name='pre_hada_norm')(x))
        return skip + x


class _Layer(nn.Module):
    """One scanned layer: a block between multi-lane hyper-connections.

    Decode skip: with `answer` [B, T] (1 on answer tokens), answer tokens skip the layers in
    config.skip_layers. A skipping token keeps its stream, writes no K/V here (other tokens do not
    see it in this layer) and gives zero q/k/v to the conv taps. The engine skips these layers for
    every decode step the same way, so training matches inference."""
    config: ModelConfig

    @nn.compact
    def __call__(self, stream, xs, mask, local_mask, rope, engram_kv, span, answer):
        site_flags, is_global, hc, layer, skipped = xs
        if local_mask is not None:
            mask = jnp.where(is_global > 0, mask, local_mask)
        keep = None
        if answer is not None:
            keep = 1 - answer * skipped
            mask = mask & (keep > 0.5)[:, None, None, :]
        B, T, n, C = stream.shape
        xf = stream.astype(jnp.float32)
        nx = quantize.a8(_rms_unit(stream.reshape(B, T, n * C)))
        h_pre = nn.sigmoid(hc['a_pre'] * (nx @ hc['phi_pre']) + hc['b_pre'] + hc['pre_off'])
        u = jnp.einsum('btn,btnc->btc', h_pre, xf)
        y = Block(self.config, name='block')(u, mask, rope, engram_kv, site_flags, layer, span, keep) - u
        h_post = 2 * nn.sigmoid(hc['a_post'] * (nx @ hc['phi_post']) + hc['b_post'] + hc['post_off'])
        h_res = _sinkhorn(hc['a_res'] * (nx @ hc['phi_res']).reshape(B, T, n, n) + hc['b_res'])
        out = jnp.einsum('btij,btjc->btic', h_res, xf) + h_post[..., None] * y[:, :, None, :]
        if keep is not None:
            out = jnp.where(keep[:, :, None, None] > 0.5, out, xf)
        return out, None


class Stack(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, x, mask, rope, engram_kv, span, answer=None):
        c = self.config
        L, n, nC = c.num_layers, c.mhc_lanes, c.mhc_lanes * c.d_model
        flags = np.zeros((L, max(1, len(c.engram_layers))), np.float32)
        for site, layer in enumerate(c.engram_layers):
            flags[layer, site] = 1.0

        local_mask, is_global = None, jnp.zeros((L,), jnp.float32)
        if c.sliding_window and c.sliding_window < x.shape[1]:
            pos = jnp.arange(x.shape[1])
            local_mask = mask & ((pos[:, None] - pos[None, :]) < c.sliding_window)[None, None]
            is_global = jnp.asarray([float(i in c.global_layers) for i in range(L)])

        lane = np.eye(n, dtype=np.float32)[np.arange(L) % n]
        hc = {
            'phi_pre': self.param('mhc_phi_pre', _init(), (L, nC, n)),
            'phi_post': self.param('mhc_phi_post', _init(), (L, nC, n)),
            'phi_res': self.param('mhc_phi_res', _init(), (L, nC, n * n)),
            'b_pre': self.param('mhc_b_pre', jinit.zeros, (L, n)),
            'b_post': self.param('mhc_b_post', jinit.zeros, (L, n)),
            'b_res': self.param('mhc_b_res', lambda *_: jnp.broadcast_to(4.0 * jnp.eye(n), (L, n, n)), (L, n, n)),
            'a_pre': self.param('mhc_a_pre', jinit.constant(0.01), (L,)),
            'a_post': self.param('mhc_a_post', jinit.constant(0.01), (L,)),
            'a_res': self.param('mhc_a_res', jinit.constant(0.01), (L,)),
            'pre_off': jnp.asarray(8 * lane - 4),
            'post_off': jnp.asarray(-4 * (1 - lane)),
        }
        stream = jnp.broadcast_to(x[:, :, None, :], (*x.shape[:2], n, x.shape[-1]))
        scan = nn.scan(nn.remat(_Layer) if c.remat else _Layer,
                       variable_axes={'params': 0, 'intermediates': 0},
                       split_rngs={'params': True}, length=L,
                       in_axes=(0,) + (nn.broadcast,) * 6)
        skipped = np.asarray([float(i in c.skip_layers) for i in range(L)], np.float32)
        if not c.skip_layers:
            answer = None
        stream, _ = scan(c, name='layers')(
            stream, (jnp.asarray(flags), is_global, hc, jnp.arange(L, dtype=jnp.int32), jnp.asarray(skipped)),
            mask, local_mask, rope, engram_kv, span, answer)
        return ZCRMSNorm(name='final_norm')(jnp.mean(stream, axis=2))


class SpindleModel(nn.Module):
    config: ModelConfig

    def setup(self):
        c = self.config
        self.embedding = nn.Embed(c.vocab_size, c.d_model)
        self.stack = Stack(c)
        tables = len(c.engram_orders) * c.engram_heads
        self.engrams = [Engram(c.d_model, tables, c.engram_slots, c.engram_sub_dim, max(c.engram_orders))
                        for _ in c.engram_layers]

    def _engram_kv(self, tokens, mask):
        c = self.config
        if not self.engrams:
            return None
        indices = engram_indices(tokens, c.engram_orders, c.engram_heads, c.engram_slots, c.engram_seed_heads)
        ngram_ok = jnp.stack([_mask_diagonal(mask, o - 1) for o in c.engram_orders
                              for _ in range(c.engram_heads)], axis=-1)
        tap_ok = jnp.stack([_mask_diagonal(mask, j * max(c.engram_orders)) for j in range(ENGRAM_CONV_TAPS)])
        pairs = [e(indices, ngram_ok, tap_ok) for e in self.engrams]
        return jnp.stack([k for k, _ in pairs]), jnp.stack([v for _, v in pairs])

    def __call__(self, tokens, mask, span=None, answer=None):
        """Logits [B, T, out_vocab], with 8-bit activations and KV as the engine runs them.

        `span` = (query, start, end) arrays switch on the span attention loss (training only).
        `answer` [B, T] (1 on answer tokens) applies decode skip (config.skip_layers) as the engine does;
        without it every token runs every layer.
        Weights are used as given: add `quantize.cq_offsets` to train for 4-bit.
        """
        c = self.config
        x = self.embedding(tokens) * math.sqrt(c.d_model)
        rope = rope_tables(c.qk_head_dim, tokens.shape[1], c.rope_theta)
        x = self.stack(x, mask, rope, self._engram_kv(tokens, mask), span, answer)
        head = self.embedding.embedding[:c.out_vocab or c.vocab_size]
        return quantize.a8(x) @ head.T


def attention_mask(tokens, window, pad_id):
    """Causal mask, limited to the last `window` tokens, with padding keys hidden. [B, 1, T, T]."""
    T = tokens.shape[1]
    pos = jnp.arange(T)
    allowed = pos[:, None] >= pos[None, :]
    if window:
        allowed &= (pos[:, None] - pos[None, :]) < window
    return allowed[None, None] & (tokens != pad_id)[:, None, None, :]
