"""Fake quantization, so training sees the same numbers as the native engine.

* Activations and the KV cache: symmetric 8-bit per row (A8 / KV8).
* Weight matrices: CQ4. Each group of 128 inputs is rotated by a Hadamard matrix, split into
  a norm (FP16) and a unit direction, and every direction value is snapped to a 16-level
  codebook. Training uses a straight-through estimator: the forward pass sees the 4-bit
  weights, the gradient flows to the FP32 weights.
"""
import jax
import jax.numpy as jnp
import numpy as np

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


def is_quantized(path, leaf):
    """Matrices the archive stores as CQ4: dense kernels, embeddings and the mHC projections."""
    key = str(getattr(path[-1], 'key', ''))
    return (key in ('kernel', 'embedding') or key.startswith('mhc_phi')) and getattr(leaf, 'ndim', 0) >= 2


def cq4_ste(params, codebook):
    """Every CQ4 matrix replaced by its 4-bit value in the forward pass (straight-through)."""
    codebook = jnp.asarray(codebook)

    def ste(path, w):
        if not is_quantized(path, w):
            return w
        # Kernels and mHC projections are stored [in, out]; their groups run along `in`.
        input_first = str(getattr(path[-1], 'key', '')) != 'embedding'
        x = jnp.swapaxes(w, -1, -2) if input_first else w
        x = x + jax.lax.stop_gradient(cq_quantize(x, codebook) - x)
        return jnp.swapaxes(x, -1, -2) if input_first else x

    return jax.tree_util.tree_map_with_path(ste, params)
