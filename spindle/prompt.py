"""The chat format the model is trained and run with.

    [<|im_start|>system\n{system}<|im_end|>\n]
    <|im_start|>user\n<tools>{tools as compact JSON}</tools>\n{query}<|im_end|>\n
    <|im_start|>assistant\n
    <tool_call>{calls as compact JSON}</tool_call><|im_end|>

An empty call list (`[]`) means "no tool fits this request".
"""
import json

IM_START, IM_END = '<|im_start|>', '<|im_end|>'
TOOLS_START, TOOLS_END = '<tools>', '</tools>'
TOOL_CALL_START, TOOL_CALL_END = '<tool_call>', '</tool_call>'


def compact(value):
    return value if isinstance(value, str) else json.dumps(value, separators=(',', ':'), ensure_ascii=False)


def render_tools(tools, system=''):
    """The part of the prompt before the request: the same for every request with these tools."""
    system = (system or '').strip()
    head = f'{IM_START}system\n{system}{IM_END}\n' if system else ''
    return f'{head}{IM_START}user\n{TOOLS_START}{compact(tools)}{TOOLS_END}\n'


def render_prompt(tools, query, system=''):
    return f'{render_tools(tools, system)}{query}{IM_END}\n{IM_START}assistant\n'


def render_target(calls):
    return f'{TOOL_CALL_START}{compact(calls)}{TOOL_CALL_END}{IM_END}'


def render_example(row):
    """(prompt, target) for a training row {query, tools, answers, [system]}."""
    return (render_prompt(row.get('tools', []), row['query'], row.get('system', '')),
            render_target(row.get('answers', [])))
