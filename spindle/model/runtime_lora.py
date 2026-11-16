"""Runtime LoRA: small FP16 updates on the attention projections, kept next to the 4-bit weights.

Plain fine-tuning barely moves the CQ4 attention weights: each optimizer step is far smaller
than the gap between two 4-bit codes, so export rounds the change away. Runtime LoRA keeps a
low-rank update A @ B beside each attention projection instead. Training adds it after the
4-bit (straight-through) weights, exactly as the native engine does at inference, and the
archive stores it in FP16, so nothing is rounded away.
"""
import jax
import jax.numpy as jnp
import numpy as np

from spindle.archive import LORA_TARGETS

RANK = 16


def _attention(params):
    return params['stack']['layers']['block']['self_attn']


def init(params, key, rank=RANK):
    """A ~ N(0, 1/in) and B = 0, so the model is unchanged at step 0."""
    attention, result = _attention(params), {}
    for name, subkey in zip(LORA_TARGETS, jax.random.split(key, len(LORA_TARGETS))):
        layers, fan_in, fan_out = attention[name]['kernel'].shape
        result[name] = {'A': jax.random.normal(subkey, (layers, fan_in, rank)) / np.sqrt(fan_in),
                        'B': jnp.zeros((layers, rank, fan_out))}
    return result


def widen(lora, key, rank):
    """The same LoRA at a higher rank: new A columns are random and new B rows are zero, so A @ B is unchanged."""
    result = {}
    for name, subkey in zip(LORA_TARGETS, jax.random.split(key, len(LORA_TARGETS))):
        A, B = jnp.asarray(lora[name]['A']), jnp.asarray(lora[name]['B'])
        layers, fan_in, old = A.shape
        extra = jax.random.normal(subkey, (layers, fan_in, rank - old)) / np.sqrt(fan_in)
        result[name] = {'A': jnp.concatenate([A, extra], axis=-1),
                        'B': jnp.concatenate([B, jnp.zeros((layers, rank - old, B.shape[-1]))], axis=1)}
    return result


def apply(params, lora):
    """Parameters with every attention kernel replaced by kernel + A @ B."""
    attention = dict(_attention(params))
    for name in LORA_TARGETS:
        attention[name] = {'kernel': attention[name]['kernel'] + lora[name]['A'] @ lora[name]['B']}
    stack = params['stack']
    block = dict(stack['layers']['block'], self_attn=attention)
    return dict(params, stack=dict(stack, layers=dict(stack['layers'], block=block)))
