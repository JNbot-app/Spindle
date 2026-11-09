"""Span attention supervision: point attention at the query text an argument copies.

A common error is copying the wrong part of the query into an argument (a value from a
neighbouring clause, a cut-off string) or leaving an argument out. When an argument value
(string or number) is written in the query, we know exactly which query tokens hold it.
For each such value we take two positions in the answer:
  * the one that predicts the argument's key (does the model write this argument?),
  * the one that predicts the value's first token (what does it copy?),
and both should attend to the value's query tokens.

The loss asks that in each layer of SPAN_LAYERS at least one head puts its attention on that
span: per layer it takes the best head's -log(attention mass on the span). Attention can
only change through the runtime LoRA (and the FP16 norms), so train it together with LoRA.
"""
import copy
import re

import numpy as np

from spindle.prompt import render_example

PER_ROW = 12                         # targets per training row (two per copied value)
SPAN_LAYERS = tuple(range(10, 20))   # the upper half of the 20 layers


def copied_values(row):
    """(call index, argument, start, end) for every argument value found in the query.

    Strings match case-insensitively; numbers match whole digit runs only (1 does not match
    inside 120). Booleans and values the query does not contain are skipped.
    """
    query = row['query'].lower()
    for c, call in enumerate(row['answers']):
        for name, value in call['arguments'].items():
            if isinstance(value, bool) or not isinstance(value, (str, int, float)) or value == '':
                continue
            if isinstance(value, str):
                pattern = re.escape(value.lower())
            else:
                pattern = r'(?<![\d.])' + re.escape(str(value)) + r'(?![\d]|\.\d)'
            found = re.search(pattern, query)
            if found:
                yield c, name, found.start(), found.end()


def _first_change(gold, tokenizer, row):
    """Index of the first answer token that differs between `gold` and `row`'s answer."""
    ids = tokenizer.encode(render_example(row)[1])
    return next((k for k, (x, y) in enumerate(zip(gold, ids)) if x != y), None)


def span_targets(row, tokenizer):
    """Up to PER_ROW (position, span start, span end) triples; positions index the full sequence.

    The key position is found by renaming the key and the value position by changing the
    value: the first answer token that changes is the one the model decides there. (Removing
    the argument instead would put the key target on the token that closes the previous
    string, which teaches the model to cut strings short.)
    """
    prompt, target = render_example(row)
    offset = 1 + len(tokenizer.encode(prompt))  # BOS + prompt
    gold = tokenizer.encode(target)
    query_at = prompt.rfind(row['query'])
    if query_at < 0:
        return []
    found = []
    for c, name, a, b in copied_values(row):
        start = 1 + len(tokenizer.encode(prompt[:query_at + a].rstrip()))
        end = max(1 + len(tokenizer.encode(prompt[:query_at + b])), start + 1)
        renamed, changed = copy.deepcopy(row), copy.deepcopy(row)
        arguments = renamed['answers'][c]['arguments']
        renamed['answers'][c]['arguments'] = {('#' if k == name else k): v for k, v in arguments.items()}
        changed['answers'][c]['arguments'][name] = '#' if row['query'][a] != '#' else '@'
        for j in (_first_change(gold, tokenizer, renamed), _first_change(gold, tokenizer, changed)):
            # Position p holds the logits that predict token p + 1.
            item = (offset + j - 1, start, end) if j else None
            if item and item not in found and len(found) < PER_ROW:
                found.append(item)
    return found


def as_arrays(found):
    """[3, PER_ROW] int32 (position, start, end); position 0 marks an empty slot."""
    result = np.zeros((3, PER_ROW), np.int32)
    for k, item in enumerate(found):
        result[:, k] = item
    return result


def span_loss(q, k, mask, query, start, end, layer):
    """Summed best-head -log(attention mass on the span) for this layer (0 outside SPAN_LAYERS).

    q [B,H,T,D] and k [B,Hkv,T,D] are the normed, rotated attention inputs; mask [B,1,T,T];
    query/start/end are [B, PER_ROW].
    """
    import jax
    import jax.numpy as jnp
    B, H, T, D = q.shape
    rows = jnp.arange(B)[:, None]
    qs = jnp.swapaxes(q[rows, :, query], 1, 2).astype(jnp.float32)                 # [B,H,S,D]
    keys = jnp.repeat(k, H // k.shape[1], axis=1).astype(jnp.float32)               # [B,H,T,D]
    allowed = mask[rows, 0, query]                                                  # [B,S,T]
    scores = jnp.einsum('bhsd,bhtd->bhst', qs, keys) / jnp.sqrt(jnp.float32(D))
    weights = jax.nn.softmax(jnp.where(allowed[:, None], scores, -1e30), axis=-1)
    positions = jnp.arange(T)
    span = (positions >= start[..., None]) & (positions < end[..., None])            # [B,S,T]
    mass = jnp.sum(weights * span[:, None], axis=-1)                                 # [B,H,S]
    best = jnp.min(-jnp.log(mass + 1e-6), axis=1)                                    # [B,S]
    valid = (query > 0) & jnp.any(span & allowed, axis=-1)
    total = jnp.sum(jnp.where(valid, best, 0.))
    return jnp.where(jnp.isin(layer, jnp.asarray(SPAN_LAYERS)), total, 0.)
