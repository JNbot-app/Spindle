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
from . import decision as decision_module, spans


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


BUCKET = 128  # rows are padded to a multiple of this, so only a few array shapes exist


class Dataset:
    """Tokenized rows. Batches are padded to the length of their longest row, rounded up to a
    multiple of BUCKET, so short rows do not pay for long ones."""

    def __init__(self, rows, tokenizer, *, max_len=512, argument_weight=1.0, no_call_weight=1.0,
                 with_spans=False, decisions=None, name='data'):
        self.pad_id, self.max_len = tokenizer.pad_id, max_len
        self.rows, skipped = [], 0
        for i, row in enumerate(rows):
            ids, weights = encode(row, tokenizer, argument_weight, no_call_weight)
            if len(ids) > max_len:
                skipped += 1
            else:
                targets = spans.span_targets(row, tokenizer) if with_spans and row['answers'] else []
                self.rows.append((ids, weights, spans.as_arrays(targets),
                                  decisions.encode(row, ids, weights) if decisions else None))
            if (i + 1) % 2000 == 0 or i + 1 == len(rows):
                print(f'  {name}: tokenized {i + 1:,}/{len(rows):,} rows', flush=True)
        if not self.rows:
            raise ValueError(f'{name}: no rows fit in {max_len} tokens')
        self.lengths = np.asarray([len(row[0]) for row in self.rows])
        if skipped:
            print(f'  {name}: skipped {skipped:,} rows longer than {max_len} tokens', flush=True)
        if with_spans:
            targets = sum(int((row[2][0] > 0).sum()) for row in self.rows)
            print(f'  {name}: {targets:,} span attention targets', flush=True)
        if decisions:
            print(f'  {name}: {sum(row[3] is not None for row in self.rows):,} rows with a call / no-call decision '
                  f'for the teacher', flush=True)
        print(f'  {name}: {len(self):,} rows, mean {self.lengths.mean():.0f} tokens, '
              f'{(self.lengths > 256).mean():.0%} over 256', flush=True)

    def __len__(self):
        return len(self.rows)

    def padded_length(self, indices):
        longest = int(self.lengths[np.asarray(indices)].max())
        return min(self.max_len, -(-longest // BUCKET) * BUCKET)

    def batch(self, indices, size, length=None):
        """Rows `indices` as arrays of `size` rows (the rest are zero-weight copies of the first row)."""
        indices = list(indices)
        length = length or self.padded_length(indices)
        tokens = np.full((size, length), self.pad_id, np.int32)
        weights = np.zeros((size, length), np.float32)
        span = np.zeros((size, 3, spans.PER_ROW), np.int32)
        teacher = np.full((size, length), self.pad_id, np.int32)
        teacher_pos, decide_pos, decide = np.zeros(size, np.int32), np.zeros(size, np.int32), np.zeros(size, np.float32)
        call_ids = np.full((size, decision_module.MAX_TOOLS), -1, np.int32)
        for k, index in enumerate(indices + indices[:1] * (size - len(indices))):
            ids, row_weights, row_spans, row_decision = self.rows[index]
            tokens[k, :len(ids)] = ids
            if k < len(indices):
                weights[k, :len(ids)], span[k] = row_weights, row_spans
                if row_decision is not None:
                    teacher_ids, decide_pos[k], calls = row_decision
                    teacher[k, :len(teacher_ids)], teacher_pos[k] = teacher_ids, len(teacher_ids) - 1
                    call_ids[k, :len(calls)], decide[k] = calls, 1.
        return {'tokens': tokens, 'weights': weights, 'teacher_tokens': teacher, 'teacher_pos': teacher_pos,
                'decide_pos': decide_pos, 'decide': decide, 'call_ids': call_ids,
                'span_query': span[:, 0], 'span_start': span[:, 1], 'span_end': span[:, 2]}

    def groups(self, indices, size):
        """Split the rows of one update into groups of `size` rows of similar length."""
        ordered = sorted(indices, key=lambda i: self.lengths[i])
        return [ordered[start:start + size] for start in range(0, len(ordered), size)]
