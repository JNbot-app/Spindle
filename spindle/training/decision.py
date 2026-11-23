"""Decision distillation: a reasoning teacher's call / no-call choice, taught to a calls-only model.

A model trained with reasoning notes (see spindle/reasoning.py) makes its decision in the first
word of the note: a tool name ("show_map: ...") or a decline ("No tool available ...").
A calls-only model makes the same decision in the token after <tool_call>: `[]` (no call) or
`[{"` (a call). For each tool-calling or no-call row the teacher reads the prompt plus
"<think>\n"; its chance of declining is

    P(decline) = P(decline words) / (P(decline words) + P(first token of any listed tool name))

and the student's two-way choice `[]` vs `[{"` at that position is pulled towards it
(cross-entropy on the two-way choice, weight `distill_weight`). Only this one decision is
distilled, so the teacher never scores answer tokens it would not write itself.
"""
import numpy as np

from spindle.prompt import THINK_START, TOOL_CALL_START
from spindle.reasoning import DECLINES

MAX_TOOLS = 64  # distinct first tokens of tool names per row (rows with more are left out)


def _single(ids, text):
    if len(ids) != 1:
        raise ValueError(f'{text!r} is not one token for this tokenizer: {ids}')
    return ids[0]


class Decisions:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.tool_call = _single(tokenizer.encode(TOOL_CALL_START), TOOL_CALL_START)
        self.no_call, self.call = _single(tokenizer.encode('[]'), '[]'), _single(tokenizer.encode('[{"'), '[{"')
        self.opening = tokenizer.encode(f'{THINK_START}\n')
        self.declines = sorted({self.first_token(text) for text in DECLINES} - {None})
        self._names = {}

    def first_token(self, text):
        """The first token of the note `text`, or None when it merges with the opening."""
        ids = self.tokenizer.encode(f'{THINK_START}\n{text}')
        return ids[len(self.opening)] if ids[:len(self.opening)] == self.opening and len(ids) > len(self.opening) else None

    def encode(self, row, ids, weights):
        """(teacher ids, student position, call token ids) for one encoded row, or None when the
        row has no clean call / no-call decision."""
        if row.get('task') in ('extract', 'classify') or row.get('teacher') is False:  # exact labels, no teacher
            return None
        answer = np.flatnonzero(weights > 0)
        if not len(answer):
            return None
        start = int(answer[0])
        if ids[start] != self.tool_call or start + 1 >= len(ids) or ids[start + 1] not in (self.no_call, self.call):
            return None
        calls = set()
        for tool in row.get('tools', []):
            name = tool.get('name', '')
            if name not in self._names:
                self._names[name] = self.first_token(f'{name}:')
            calls.add(self._names[name])
        if not calls or None in calls or calls & set(self.declines) or len(calls) > MAX_TOOLS:
            return None
        return list(ids[:start]) + self.opening, start, sorted(calls)
