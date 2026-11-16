"""Tool-calling and no-call rows from public function-calling datasets.

Every source yields rows {query, tools, answers, task, source}; task is 'tools' when the answer
has a call and 'no_call' when it is empty. Only the first user turn of a conversation is used.
"""
import ast
import json
import re

from spindle import hub
from spindle.convert import call, json_type, tool


def row(query, tools, answers, source):
    return {'query': query.strip(), 'tools': tools, 'answers': answers,
            'task': 'tools' if answers else 'no_call', 'source': source}


def _valid(tools, answers):
    names = {t['name'] for t in tools}
    return (tools and len(names) == len(tools) and all(isinstance(a['arguments'], dict) for a in answers)
            and all(a['name'] in names for a in answers))


# ---------------------------------------------------------------- xLAM
def xlam():
    """Salesforce/xlam-function-calling-60k. Gated: accept its terms on the Hub and log in first."""
    try:
        data = json.load(open(hub.hf_file('Salesforce/xlam-function-calling-60k', 'xlam_function_calling_60k.json')))
    except Exception as problem:
        print(f'  xlam: skipped ({type(problem).__name__}). Accept the dataset terms on huggingface.co and run '
              f'`huggingface-cli login` to include it.', flush=True)
        return
    for item in data:
        tools = []
        for t in json.loads(item['tools']):
            parameters = t.get('parameters') or {}
            properties = {}
            for name, spec in parameters.items():
                properties[name] = {'type': json_type(spec.get('type', 'str')) or 'string',
                                    'description': spec.get('description', '')}
            required = [n for n, spec in parameters.items()
                        if 'default' not in spec and 'optional' not in str(spec.get('type', ''))]
            tools.append(tool(t['name'], t.get('description'),
                              {'type': 'object', 'properties': properties, 'required': required}))
        answers = [call(a['name'], a['arguments']) for a in json.loads(item['answers'])]
        if _valid(tools, answers):
            yield row(item['query'], tools, answers, 'xlam')


# ---------------------------------------------------------------- ToolACE
def _split_calls(text):
    """'[a(x=1), b()]' -> ['a(x=1)', 'b()'], respecting quotes and nesting."""
    text = text.strip()
    if not (text.startswith('[') and text.endswith(']')):
        return None
    parts, depth, quote, start, i = [], 0, '', 1, 1
    while i < len(text) - 1:
        c = text[i]
        if quote:
            if c == '\\':
                i += 1
            elif c == quote:
                quote = ''
        elif c in '"\'':
            quote = c
        elif c in '([{':
            depth += 1
        elif c in ')]}':
            depth -= 1
            if depth == 0 and c == ')':
                parts.append(text[start:i + 1].strip())
                start = i + 1
                while start < len(text) - 1 and text[start] in ', \n':
                    start += 1
                i = start - 1
        i += 1
    return parts if parts and start >= len(text) - 1 and depth == 0 else None


def _parse_call(text):
    name, _, arguments = text.partition('(')
    tree = ast.parse(f'f({arguments}', mode='eval').body
    if tree.args:
        raise ValueError('positional argument')
    return call(name.strip(), {k.arg: ast.literal_eval(k.value) for k in tree.keywords})


def toolace():
    marker = 'Here is a list of functions in JSON format that you can invoke:\n'
    for item in json.load(open(hub.hf_file('Team-ACE/ToolACE', 'data.json'))):
        turns = item['conversations']
        if marker not in item['system'] or len(turns) < 2 or turns[0]['from'] != 'user':
            continue
        try:
            listed = json.JSONDecoder().raw_decode(item['system'].split(marker, 1)[1].strip())[0]
            tools = [tool(t['name'], t.get('description'), t.get('parameters')) for t in listed]
            parts = _split_calls(turns[1]['value'])
            answers = [_parse_call(p) for p in parts] if parts else []
        except (ValueError, SyntaxError, KeyError, TypeError):
            continue
        if _valid(tools, answers):
            yield row(turns[0]['value'], tools, answers, 'toolace')


# ---------------------------------------------------------------- Glaive
def _refuses(reply):
    """True when the assistant says it cannot do the request (no listed tool fits). Replies that ask
    for more details are not used as "no call" rows: there the right tool is in the list."""
    return bool(re.search(r"sorry|unable|can't|cannot|don't have|apolog", reply[:220], re.I))


def glaive():
    decoder = json.JSONDecoder()
    for item in json.load(open(hub.hf_file('glaiveai/glaive-function-calling-v2',
                                           'glaive-function-calling-v2.json'))):
        text, tools = item['system'].partition('Use them if required -')[2], []
        try:
            position = text.find('{')
            while position >= 0:
                spec, end = decoder.raw_decode(text, position)
                tools.append(tool(spec['name'], spec.get('description'), spec.get('parameters')))
                position = text.find('{', end)
        except (ValueError, KeyError):
            continue
        turns = re.split(r'\n\n\n(?=USER: |ASSISTANT: |FUNCTION RESPONSE: )', item['chat'].strip())
        if not tools or not turns or not turns[0].startswith('USER: '):
            continue
        answers = []
        if len(turns) > 1 and '<functioncall>' not in turns[1] and not _refuses(turns[1]):
            continue  # a follow-up question or small talk: a tool may fit, so it is not a clean "no call"
        for turn in turns[1:]:
            if not turn.startswith('ASSISTANT: '):
                break
            if '<functioncall>' in turn:
                body = turn.split('<functioncall>', 1)[1].replace('<|endoftext|>', '').strip()
                found = re.fullmatch(r'\{"name":\s*"([^"]+)",\s*"arguments":\s*\'(.*)\'\s*\}', body, re.S)
                try:
                    answers = [call(found.group(1), json.loads(found.group(2)))]
                except (AttributeError, ValueError):
                    answers = None
                break
        if answers is not None and _valid(tools, answers):
            yield row(turns[0][len('USER: '):], tools, answers, 'glaive')


# ---------------------------------------------------------------- When2Call
def when2call():
    """Requests no listed tool can serve: the answer is no call."""
    for item in hub.hf_jsonl('nvidia/When2Call', 'train/when2call_train_sft.jsonl'):
        user, reply = item['messages'][0], item['messages'][1]
        if user['role'] != 'user' or 'TOOLCALL' in reply['content'].upper() or '"name"' in reply['content']:
            continue
        if not _refuses(reply['content']):
            continue  # asks for a missing value: a listed tool fits, so "no call" would teach declining
        try:
            tools = [tool(t['name'], t.get('description'), t.get('parameters'))
                     for t in map(json.loads, item['tools'])]
        except (ValueError, KeyError):
            continue
        if _valid(tools, []):
            yield row(user['content'], tools, [], 'when2call')


SOURCES = {'xlam': xlam, 'toolace': toolace, 'glaive': glaive, 'when2call': when2call}
