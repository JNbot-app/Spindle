"""Reasoning notes: one short line the model writes before its calls.

    <think>
    show_map: query 'Piazza Navona in Rome' from query.
    </think>
    <tool_call>[...]</tool_call>

Writing the values down first makes the copied text more exact. The notes are built from a
training row's answer, so no extra labelling is needed. A row may also carry its own `reasoning`.
"""
import random

MAX_VALUE = 60   # longer values are named, not quoted
MAX_NOTE = 320
DECLINES = ('No tool available for this request.', 'No listed tool fits this request.',
            'None of the tools can do this.', 'No tool available for what the user asks.')


def decline_note(seed=''):
    return random.Random(str(seed)).choice(DECLINES)


def _part(name, value, query):
    if isinstance(value, (dict, list)):
        return f'{name} given'
    text = str(value).lower() if isinstance(value, bool) else str(value)
    copied = isinstance(value, (str, int, float)) and not isinstance(value, bool) and text.lower() in query.lower()
    if len(text) > MAX_VALUE:
        return f'{name} from query' if copied else f'{name} given'
    return f"{name} '{text}'" + (' from query' if copied else '')


def note(row):
    """The reasoning line for a row {query, answers}."""
    if not row['answers']:
        return decline_note(row['query'])
    lines = []
    for call in row['answers']:
        parts = [_part(name, value, row['query']) for name, value in call['arguments'].items()]
        lines.append(f"{call['name']}: {'; '.join(parts) if parts else 'no arguments'}.")
    text = ' '.join(lines)
    return text if len(text) <= MAX_NOTE else text[:MAX_NOTE].rsplit(' ', 1)[0] + ' ...'
