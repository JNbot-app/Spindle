"""The chat format the model is trained and run with.

    [<|im_start|>system\n{system}<|im_end|>\n]
    <|im_start|>user\n<tools>{tools as compact JSON}</tools>\n{query}<|im_end|>\n
    <|im_start|>assistant\n
    [<think>\n{reasoning}\n</think>\n]<tool_call>{calls as compact JSON}</tool_call><|im_end|>

An empty call list (`[]`) means "no tool fits this request". The reasoning note is optional
(see reasoning.py).

The model knows the current date as a short fact at the start of the system prompt:
`date: 2024-06-06 Thu 09:58`. A system prompt that gives a timestamp in another way
("Current date and time: 2024-06-06T09:58:41") is rewritten to that fact (see `date_fact`).
"""
import json
import re
from datetime import datetime

IM_START, IM_END = '<|im_start|>', '<|im_end|>'
TOOLS_START, TOOLS_END = '<tools>', '</tools>'
TOOL_CALL_START, TOOL_CALL_END = '<tool_call>', '</tool_call>'
THINK_START, THINK_END = '<think>', '</think>'


def compact(value):
    return value if isinstance(value, str) else json.dumps(value, separators=(',', ':'), ensure_ascii=False)


_STAMP = re.compile(r'(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})(?::\d{2})?')
_DATE_KEY = re.compile(r'(?<![A-Za-z])date:')
_WEEKDAY_LINE = re.compile(r'^\W*day of (the )?week\b[^\n]*$', re.IGNORECASE)


def date_fact(system):
    """The system prompt with its timestamp as the date fact the model was trained on.

    The line with the timestamp (and a line that only names the weekday) is replaced by
    `date: YYYY-MM-DD Ddd HH:MM`; other lines are kept after it. Text without a timestamp,
    or with its own `date:` fact, is returned unchanged."""
    system = (system or '').strip()
    match = _STAMP.search(system)
    if not match or _DATE_KEY.search(system):
        return system
    try:
        moment = datetime(*(int(part) for part in match.groups()))
    except ValueError:
        return system
    rest = [line.strip() for line in system.split('\n')
            if line.strip() and not _STAMP.search(line) and not _WEEKDAY_LINE.match(line)]
    return '; '.join([moment.strftime('date: %Y-%m-%d %a %H:%M')] + rest)


def render_tools(tools, system=''):
    """The part of the prompt before the request: the same for every request with these tools."""
    system = date_fact(system)
    head = f'{IM_START}system\n{system}{IM_END}\n' if system else ''
    return f'{head}{IM_START}user\n{TOOLS_START}{compact(tools)}{TOOLS_END}\n'


def render_prompt(tools, query, system=''):
    return f'{render_tools(tools, system)}{query}{IM_END}\n{IM_START}assistant\n'


def render_target(calls, reasoning=''):
    think = f'{THINK_START}\n{reasoning.strip()}\n{THINK_END}\n' if reasoning else ''
    return f'{think}{TOOL_CALL_START}{compact(calls)}{TOOL_CALL_END}{IM_END}'


def render_example(row):
    """(prompt, target) for a training row {query, tools, answers, [system], [reasoning]}."""
    return (render_prompt(row.get('tools', []), row['query'], row.get('system', '')),
            render_target(row.get('answers', []), row.get('reasoning', '')))
