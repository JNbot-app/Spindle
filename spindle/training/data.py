"""Training data: JSONL rows -> padded token arrays, loss weights and span targets.

One row per line:
    {"query": "...", "tools": [{name, description, parameters}, ...],
     "answers": [{"name": "...", "arguments": {...}}, ...], "system": "..." (optional)}
An empty `answers` list teaches the model to make no call.

Only answer tokens are trained. Tokens inside argument values get `argument_weight`, and
rows without a call get `no_call_weight` (so the rarer "no call" decision is not drowned out).
"""
import json

import numpy as np

from spindle.prompt import render_example
from . import spans


def read_rows(path):
    rows = []
    with open(path, encoding='utf-8') as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get('query'), str) or not isinstance(row.get('answers'), list):
                raise ValueError(f'{path}:{number}: each row needs a "query" string and an "answers" list')
            rows.append(row)
    return rows


def argument_spans(target):
    """Character spans of every argument value inside the rendered answer."""
    start = target.index('<tool_call>') + len('<tool_call>')
    text = target[start:target.rindex('</tool_call>')]
    decoder = json.JSONDecoder()
    found = []

    def skip(i):
        while text[i].isspace():
            i += 1
        return i

    def walk(i, path):
        i = skip(i)
        if text[i] in '{[':
            closing, n = '}' if text[i] == '{' else ']', 0
            i = skip(i + 1)
            if text[i] == closing:
                return i + 1
            while True:
                if closing == '}':
                    key, i = decoder.raw_decode(text, skip(i))
                    i = skip(i) + 1  # the colon
                    i = walk(i, (*path, key))
                else:
                    i = walk(i, (*path, n))
                    n += 1
                i = skip(i)
                if text[i] == closing:
                    return i + 1
                i += 1  # the comma
        value, end = decoder.raw_decode(text, i)
        # Values inside a top-level call's "arguments" object (path: call index, 'arguments', ...).
        if len(path) >= 3 and isinstance(path[0], int) and path[1] == 'arguments':
            a, b = (i + 1, end - 1) if isinstance(value, str) else (i, end)
            if a < b:
                found.append((start + a, start + b))
        return end

    walk(0, ())
    return found


def encode(row, tokenizer, argument_weight=1.0, no_call_weight=1.0):
    """(token ids, loss weights) for one row: BOS + prompt + answer + EOS."""
    prompt, target = render_example(row)
    prompt_ids = tokenizer.encode(prompt)
    pieces = tokenizer.encode_with_offsets(target)
    value_spans = argument_spans(target) if argument_weight != 1 else []
    weights = [argument_weight if any(a < e and b > s for s, e in value_spans) else 1.0 for _, a, b in pieces]
    ids = [tokenizer.bos_id] + prompt_ids + [t for t, _, _ in pieces] + [tokenizer.eos_id]
    weights = np.asarray([0.0] * (1 + len(prompt_ids)) + weights + [1.0], np.float32)
    if not row['answers']:
        weights *= no_call_weight
    return np.asarray(ids, np.int32), weights


class Dataset:
    """All rows tokenized and padded to one length, so every batch has the same shape."""

    def __init__(self, rows, tokenizer, *, max_len=512, argument_weight=1.0, no_call_weight=1.0,
                 with_spans=False, name='data'):
        encoded, skipped = [], 0
        for i, row in enumerate(rows):
            ids, weights = encode(row, tokenizer, argument_weight, no_call_weight)
            if len(ids) > max_len:
                skipped += 1
                continue
            targets = spans.span_targets(row, tokenizer) if with_spans and row['answers'] else []
            encoded.append((ids, weights, spans.as_arrays(targets)))
            if (i + 1) % 2000 == 0 or i + 1 == len(rows):
                print(f'  {name}: tokenized {i + 1:,}/{len(rows):,} rows', flush=True)
        if not encoded:
            raise ValueError(f'{name}: no rows fit in {max_len} tokens')
        if skipped:
            print(f'  {name}: skipped {skipped:,} rows longer than {max_len} tokens', flush=True)
        length = min(max_len, -(-max(len(ids) for ids, _, _ in encoded) // 64) * 64)
        self.tokens = np.full((len(encoded), length), tokenizer.pad_id, np.int32)
        self.weights = np.zeros((len(encoded), length), np.float32)
        self.spans = np.stack([s for _, _, s in encoded])
        for i, (ids, weights, _) in enumerate(encoded):
            self.tokens[i, :len(ids)], self.weights[i, :len(ids)] = ids, weights
        if with_spans:
            count = int((self.spans[:, 0] > 0).sum())
            print(f'  {name}: {count:,} span attention targets in {int((self.spans[:, 0, 0] > 0).sum()):,} rows',
                  flush=True)
        print(f'  {name}: {len(self):,} rows, padded to {length} tokens', flush=True)

    def __len__(self):
        return len(self.tokens)

    def batch(self, indices, size):
        """Rows `indices`, padded to `size` rows with zero-weight copies of the first row."""
        indices = np.asarray(indices)
        fill = np.concatenate([indices, np.repeat(indices[:1], size - len(indices))])
        weights = self.weights[fill].copy()
        span = self.spans[fill].copy()
        weights[len(indices):] = 0
        span[len(indices):] = 0
        return {'tokens': self.tokens[fill], 'weights': weights,
                'span_query': span[:, 0], 'span_start': span[:, 1], 'span_end': span[:, 2]}
