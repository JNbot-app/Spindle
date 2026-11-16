"""Benchmark sets, each converted to the same case format.

A case is {id, query, tools, expected, kind, [system], [require_call]}:
  kind 'calls'    expected = the exact list of calls, in order
  kind 'ast'      expected = BFCL ground truth: [{name: {argument: [accepted values]}}], any order
  kind 'fields'   expected = list of calls; scored as field F1 over (tool, argument, value)
  kind 'decision' expected = True (a call is needed) or False (no call)
None of these sets is used for training.
"""
import json
import random

from spindle import hub
from spindle.convert import call, tool

GORILLA = ('https://raw.githubusercontent.com/ShishirPatil/gorilla/main/berkeley-function-call-leaderboard/'
           'bfcl_eval/data/')
SGD = 'https://raw.githubusercontent.com/google-research-datasets/dstc8-schema-guided-dialogue/master/test/'
SNIPS = 'https://raw.githubusercontent.com/MiuLab/SlotGated-SLU/master/data/snips/test/'


# ---------------------------------------------------------------- tool calling
def mobile_actions():
    cases = []
    rows = [r for r in hub.hf_jsonl('google/mobile-actions', 'dataset.jsonl') if r['metadata'] == 'eval']
    for i, row in enumerate(rows):
        messages = {m['role']: m for m in row['messages']}
        tools = [tool(t['function']['name'], t['function'].get('description'), t['function'].get('parameters'))
                 for t in row['tools']]
        system = messages['developer']['content'].split('\nYou are a model')[0].strip()
        expected = [call(c['function']['name'], c['function']['arguments'])
                    for c in messages['assistant'].get('tool_calls') or []]
        cases.append({'id': f'mobile_actions_{i}', 'query': messages['user']['content'], 'system': system,
                      'tools': tools, 'expected': expected, 'kind': 'calls'})
    return cases


def droidcall():
    cases = []
    for i, row in enumerate(hub.hf_jsonl('mllmTeam/DroidCall', 'DroidCall_test.jsonl')):
        tools = []
        for t in row['tools']:
            arguments = t.get('arguments') or {}
            schema = {'type': 'object',
                      'properties': {k: {'type': v.get('type', 'str'), 'description': v.get('description', '')}
                                     for k, v in arguments.items()},
                      'required': [k for k, v in arguments.items() if v.get('required')]}
            tools.append(tool(t['name'], t.get('description'), schema))
        cases.append({'id': f'droidcall_{i}', 'query': row['query'], 'tools': tools,
                      'expected': [call(a['name'], a['arguments']) for a in row['answers']], 'kind': 'calls'})
    return cases


BFCL_CALL = ('simple_python', 'simple_java', 'simple_javascript', 'multiple', 'parallel', 'parallel_multiple',
             'live_simple', 'live_multiple', 'live_parallel', 'live_parallel_multiple')
BFCL_DECISION = {'irrelevance': False, 'live_irrelevance': False, 'live_relevance': True}


def bfcl():
    def rows(name):
        return hub.read_jsonl(hub.url_file(f'{GORILLA}{name}.json'))

    cases = []
    for category in (*BFCL_CALL, *BFCL_DECISION):
        answers = ({a['id']: a['ground_truth'] for a in rows(f'possible_answer/BFCL_v4_{category}')}
                   if category in BFCL_CALL else {})
        strings_only = category in ('simple_java', 'simple_javascript')
        for row in rows(f'BFCL_v4_{category}'):
            turns = row['question'][0]
            system = ' '.join(m['content'] for m in turns if m['role'] == 'system')
            query = '\n'.join(m['content'] for m in turns if m['role'] == 'user')
            tools = [tool(f['name'], f.get('description'), f.get('parameters'), strings_only) for f in row['function']]
            case = {'id': row['id'], 'query': query, 'tools': tools, 'group': category}
            if system:
                case['system'] = system
            if category in BFCL_CALL:
                case.update(expected=answers[row['id']], kind='ast')
            else:
                case.update(expected=BFCL_DECISION[category], kind='decision')
            cases.append(case)
    return cases


def when2call():
    cases = []
    for row in hub.hf_jsonl('nvidia/When2Call', 'test/when2call_test_mcq.jsonl'):
        tools = []
        for text in row['tools']:
            t = json.loads(text)
            tools.append(tool(t['name'], t.get('description'), t.get('parameters')))
        cases.append({'id': f'when2call_{row["uuid"]}', 'query': row['question'], 'tools': tools,
                      'expected': row['correct_answer'] == 'tool_call', 'kind': 'decision',
                      'group': row['correct_answer']})
    return cases


def tool_removed():
    """Mobile Actions single-call rows without the needed tool: the right answer is no call."""
    cases = []
    for case in mobile_actions():
        if len(case['expected']) == 1:
            needed = case['expected'][0]['name']
            cases.append({**case, 'id': case['id'].replace('mobile_actions', 'tool_removed'),
                          'tools': [t for t in case['tools'] if t['name'] != needed],
                          'expected': False, 'kind': 'decision'})
    return cases


# ---------------------------------------------------------------- extraction
def _words(name):
    return name.replace('_', ' ')


def _snips_rows():
    texts, tags, labels = (hub.url_file(SNIPS + name).read_text(encoding='utf-8').splitlines()
                           for name in ('seq.in', 'seq.out', 'label'))
    rows = []
    for text, tag, label in zip(texts, tags, labels):
        slots, current = [], None
        for word, t in zip(text.split(), tag.split()):
            if t.startswith('B-'):
                current = [t[2:], [word]]
                slots.append(current)
            elif t.startswith('I-') and current and current[0] == t[2:]:
                current[1].append(word)
            else:
                current = None
        arguments = {}
        for name, words in slots:
            arguments.setdefault(name, ' '.join(words))
        rows.append((' '.join(text.split()), label.strip(), arguments))
    return rows


def _snips(seven_way):
    rows = _snips_rows()
    slots = {}
    for _, label, arguments in rows:
        slots.setdefault(label, set()).update(arguments)
    tools = {label: tool(label, f'The user wants to: {label}.',
                         {'type': 'object', 'properties': {s: {'type': 'string', 'description': _words(s)}
                                                           for s in sorted(names)}})
             for label, names in sorted(slots.items())}
    return [{'id': f'snips_{"7way" if seven_way else "gold"}_{i}', 'query': text,
             'tools': list(tools.values()) if seven_way else [tools[label]],
             'expected': [call(label, arguments)], 'kind': 'fields', 'require_call': True}
            for i, (text, label, arguments) in enumerate(rows)]


def snips_gold():
    """SNIPS with only the right intent's tool given: pure slot extraction."""
    return _snips(False)


def snips_7way():
    """SNIPS with all seven intents as tools: pick the intent, then extract."""
    return _snips(True)


def dstc8(count=1813, seed=0):
    """User turns of the DSTC8 schema-guided test dialogues that state at least one slot value."""
    def build():
        services = {s['service_name']: s for s in json.loads(hub.url_file(SGD + 'schema.json').read_text())}

        def tools_of(service):
            slots = {s['name']: s for s in service['slots']}
            tools = []
            for intent in service['intents']:
                names = [*intent['required_slots'], *intent['optional_slots']]
                properties = {}
                for name in names:
                    slot = slots[name]
                    properties[name] = {'type': 'string', 'description': slot['description']}
                    if slot['is_categorical'] and slot['possible_values']:
                        properties[name]['enum'] = slot['possible_values']
                tools.append(tool(intent['name'], intent['description'],
                                  {'type': 'object', 'properties': properties}))
            return tools

        found = []
        for number in range(1, 35):
            dialogues = json.loads(hub.url_file(f'{SGD}dialogues_{number:03d}.json').read_text())
            for dialogue in dialogues:
                for index, turn in enumerate(dialogue['turns']):
                    if turn['speaker'] != 'USER' or len(turn['frames']) != 1:
                        continue
                    frame = turn['frames'][0]
                    intent = frame['state']['active_intent']
                    tools = tools_of(services[frame['service']])
                    allowed = next((t['parameters']['properties'] for t in tools if t['name'] == intent), None)
                    arguments = {a['slot']: a['values'][0] for a in frame['actions']
                                 if a['act'] == 'INFORM' and a['values'] and allowed and a['slot'] in allowed}
                    if arguments:
                        found.append({'id': f'dstc8_{dialogue["dialogue_id"]}_{index}', 'query': turn['utterance'],
                                      'tools': tools, 'expected': [call(intent, arguments)], 'kind': 'fields',
                                      'require_call': True, 'group': frame['service']})
        random.Random(seed).shuffle(found)
        return found[:count]

    return hub.cached(f'bench/dstc8-{count}-{seed}', build)


# ---------------------------------------------------------------- classification
def _classification(name, rows, tool_name, description, field, labels, count, seed=0):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    spec = tool(tool_name, description, {'type': 'object', 'required': [field], 'properties': {
        field: {'type': 'string', 'enum': labels}}})
    return [{'id': f'{name}_{i}', 'query': text, 'tools': [spec], 'expected': [call(tool_name, {field: labels[label]})],
             'kind': 'calls', 'require_call': True} for i, (text, label) in enumerate(rows[:count])]


def rotten_tomatoes():
    rows = hub.hf_parquet('cornell-movie-review-data/rotten_tomatoes', 'test.parquet')
    return _classification('rotten_tomatoes', ((r['text'], r['label']) for r in rows), 'classify_sentiment',
                           'Classify the sentiment of a movie review.', 'sentiment', ['negative', 'positive'], 1066)


DBPEDIA = ['company', 'educational institution', 'artist', 'athlete', 'office holder', 'mean of transportation',
           'building', 'natural place', 'village', 'animal', 'plant', 'album', 'film', 'written work']


def dbpedia():
    def build():
        rows = hub.hf_parquet('fancyzhx/dbpedia_14', 'dbpedia_14/test-00000-of-00001.parquet')
        rows = [r for r in rows if len(r['content']) < 400]
        random.Random(0).shuffle(rows)
        return [{'text': r['content'].strip(), 'label': r['label']} for r in rows[:1000]]

    rows = hub.cached('bench/dbpedia-1000', build)
    return _classification('dbpedia', ((r['text'], r['label']) for r in rows), 'classify_topic',
                           'Decide what kind of thing a short encyclopedia text describes.', 'topic', DBPEDIA, 1000)


SETS = {
    'mobile_actions': mobile_actions, 'droidcall': droidcall, 'bfcl': bfcl,
    'when2call': when2call, 'tool_removed': tool_removed,
    'dstc8': dstc8, 'snips_gold': snips_gold, 'snips_7way': snips_7way,
    'rotten_tomatoes': rotten_tomatoes, 'dbpedia': dbpedia,
}


def load(name, limit=0, seed=0):
    """The cases of one set; `limit` > 0 takes a fixed random sample of that size."""
    cases = SETS[name]()
    if limit and len(cases) > limit:
        cases = random.Random(seed).sample(cases, limit)
    return cases
