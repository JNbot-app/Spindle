"""Fake quantization, so training sees the same numbers as the native engine.

* Activations and the KV cache: symmetric 8-bit per row (A8 / KV8).
* Weight matrices: CQ at 4 or 2 bits. Each group of 128 inputs is rotated by a Hadamard matrix,
  split into a norm (FP16) and a unit direction, and every direction value is snapped to a
  16-level (4-bit) or 4-level (2-bit) codebook. Or E8 (about 2 bits): blocks of 8 rotated values
  snap to E8 lattice points with a fitted gain per group (spindle.lattice). Training uses a straight-through estimator: the
  forward pass sees the rounded weights, the gradient flows to the FP32 weights. The rounding
  offsets (`cq_offsets`) are made once per update and added to the weights in every forward pass.
"""
import jax
import jax.numpy as jnp
import numpy as np

from spindle import lattice

GROUP = 128


def _symmetric(x, bits=8):
    qmax = 2 ** (bits - 1) - 1
    xf = x.astype(jnp.float32)
    absmax = jnp.max(jnp.abs(xf), axis=-1, keepdims=True)
    scale = jnp.where(absmax > 0, absmax / qmax, 1.0)
    q = (jnp.clip(jnp.round(xf / scale), -qmax - 1, qmax) * scale).astype(x.dtype)
    return x + jax.lax.stop_gradient(q - x)


def a8(x):
    """8-bit activations: one scale per row (last axis)."""
    return _symmetric(x)


def a8_per_head(x):
    """8-bit attention inputs and KV cache: one scale per head and position ([B, H, T, D])."""
    return _symmetric(x)


def hadamard(group=GROUP):
    h = np.array([[1.0]], np.float32)
    while h.shape[0] < group:
        h = np.block([[h, h], [h, -h]])
    return (h / np.sqrt(group)).astype(np.float32)


def cq_quantize(w, codebook, group=GROUP):
    """Dequantized CQ weights for w [..., in] (groups run along the last axis)."""
    D = w.shape[-1]
    pad = (-D) % group
    wp = jnp.pad(w, [(0, 0)] * (w.ndim - 1) + [(0, pad)]) if pad else w
    rotated = wp.reshape(*wp.shape[:-1], -1, group).astype(jnp.float32) @ jnp.asarray(hadamard(group))
    norm = jnp.sqrt(jnp.sum(rotated ** 2, axis=-1, keepdims=True))
    unit = (rotated / jnp.maximum(norm, 1e-12)).reshape(-1)
    pos = jnp.clip(jnp.searchsorted(codebook, unit), 1, codebook.shape[0] - 1)
    left, right = codebook[pos - 1], codebook[pos]
    snapped = jnp.where(jnp.abs(unit - left) <= jnp.abs(unit - right), left, right)
    norm = norm.astype(jnp.float16).astype(jnp.float32)
    w_q = ((snapped.reshape(rotated.shape) * norm) @ jnp.asarray(hadamard(group))).reshape(wp.shape)
    return w_q[..., :D].astype(w.dtype)


def e8_quantize(w, group=GROUP):
    """Dequantized E8 weights for w [..., in] (groups run along the last axis), as spindle.archive stores them."""
    D = w.shape[-1]
    pad = (-D) % group
    wp = jnp.pad(w, [(0, 0)] * (w.ndim - 1) + [(0, pad)]) if pad else w
    h = jnp.asarray(hadamard(group))
    rotated = wp.reshape(*wp.shape[:-1], -1, group).astype(jnp.float32) @ h
    points, gain = lattice.quantize_groups(rotated, jnp)
    return ((points * gain) @ h).reshape(wp.shape)[..., :D].astype(w.dtype)


def is_quantized(path, leaf):
    """Matrices the archive stores as CQ: dense kernels, embeddings and the mHC projections."""
    key = str(getattr(path[-1], 'key', ''))
    return (key in ('kernel', 'embedding') or key.startswith('mhc_phi')) and getattr(leaf, 'ndim', 0) >= 2


def keeps_4bit(path):
    """Matrices a 2-bit model keeps at 4 bits, as Needle 3 does: the token embedding and the mHC mixing
    matrices (an error there reaches every token or every layer). Engram tables stay 2-bit."""
    keys = [str(getattr(p, 'key', '')) for p in path]
    return keys[0] == 'embedding' or keys[-1].startswith('mhc_phi')


def cq_offsets(params, codebook, wide_codebook=None, e8=False):
    """Rounded minus FP32 value for every CQ matrix (0 for other weights), with no gradient.

    `params + offsets` is the straight-through estimator: the forward pass sees the rounded weights,
    and the gradient reaches the FP32 weights unchanged. Making the offsets once per update (not
    in every micro-batch) gives the same result and saves most of the rounding time.

    The codebook sets the width: 16 values for 4 bits, 4 for 2 bits. With `wide_codebook` (the 4-bit
    one), the matrices `keeps_4bit` names use it instead. With `e8`, all other matrices are rounded
    to E8 lattice points instead of `codebook`."""
    codebook = jnp.asarray(codebook)
    wide = codebook if wide_codebook is None else jnp.asarray(wide_codebook)

    def offset(path, w):
        if not is_quantized(path, w):
            return jnp.zeros((), w.dtype)
        # Kernels and mHC projections are stored [in, out]; their groups run along `in`.
        input_first = str(getattr(path[-1], 'key', '')) != 'embedding'
        x = jnp.swapaxes(w, -1, -2) if input_first else w
        if keeps_4bit(path) or not e8:
            q = cq_quantize(x, wide if keeps_4bit(path) else codebook)
        else:
            q = e8_quantize(x)
        delta = jax.lax.stop_gradient(q - x)
        return jnp.swapaxes(delta, -1, -2) if input_first else delta

    return jax.tree_util.tree_map_with_path(offset, params)
