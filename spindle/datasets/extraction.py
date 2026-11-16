"""Extraction rows: slot filling (MASSIVE, ATIS), entity lists (Few-NERD, Pile-NER) and
generated records (records.py). Every row has task 'extract' and always has a call.
"""
import csv
import io
import json
import random
import re
import tarfile

import pyarrow.parquet as pq

from spindle import hub
from spindle.convert import call, tool
from . import records

MASSIVE_URL = 'https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz'


def row(query, tools, answers, source):
    return {'query': query.strip(), 'tools': tools, 'answers': answers, 'task': 'extract', 'source': source}


def words(name):
    return re.sub(r'[_.\-/]+', ' ', name).strip()


def _slot_rows(examples, source, seed, extra_fields=5):
    """(text, intent, {slot: value}) -> rows. The tools are the right intent plus a few others. An
    intent's fields come from the slots it has anywhere in the data: the needed ones plus a few
    more, so the schema differs from row to row and stays short."""
    rng = random.Random(seed)
    slots = {}
    for _, intent, arguments in examples:
        slots.setdefault(intent, set()).update(arguments)

    def spec(intent, needed=()):
        spare = sorted(slots[intent] - set(needed))
        names = [*needed, *rng.sample(spare, min(len(spare), rng.randint(1, extra_fields)))]
        rng.shuffle(names)
        return tool(intent, f'{words(intent).capitalize()}.', {'type': 'object', 'properties': {
            s: {'type': 'string', 'description': words(s).capitalize()} for s in names}})

    for text, intent, arguments in examples:
        if not arguments:
            continue
        related = [i for i in slots if i != intent and i.split('_')[0] == intent.split('_')[0]]
        others = rng.sample(related, min(len(related), rng.randint(0, 2)))
        rest = [i for i in slots if i != intent and i not in others]
        others += rng.sample(rest, min(len(rest), rng.randint(0, 3)))
        listed = [spec(intent, arguments), *(spec(i) for i in others)]
        rng.shuffle(listed)
        yield row(text, listed, [call(intent, arguments)], source)


def massive():
    """Amazon MASSIVE, English: 60 intents with slot annotations like '[time : five am]'."""
    def build():
        with tarfile.open(hub.url_file(MASSIVE_URL)) as archive:
            member = next(m for m in archive.getmembers() if m.name.endswith('en-US.jsonl'))
            lines = archive.extractfile(member).read().decode('utf-8').splitlines()
        return [json.loads(line) for line in lines]

    examples = []
    for item in hub.cached('data/massive-en', build):
        if item['partition'] != 'train':
            continue
        arguments = {}
        for name, value in re.findall(r'\[([a-z_]+) : ([^\]]+)\]', item['annot_utt']):
            arguments.setdefault(name, value.strip())
        examples.append((item['utt'], item['intent'], arguments))
    yield from _slot_rows(examples, 'massive', 0)


def atis():
    examples = []
    text = open(hub.hf_file('tuetschek/atis', 'atis_train.csv'), encoding='utf-8').read()
    for item in csv.DictReader(io.StringIO(text)):
        if '+' in item['intent']:
            continue
        arguments, current = {}, None
        for word, tag in zip(item['text'].split(), item['slots'].split()):
            name = tag[2:].replace('.', '_')
            if tag.startswith('B-'):
                current = name if name not in arguments else None
                if current:
                    arguments[current] = word
            elif tag.startswith('I-') and current == name:
                arguments[current] += ' ' + word
            else:
                current = None
        examples.append((item['text'], item['intent'], arguments))
    yield from _slot_rows(examples, 'atis', 1)


# ---------------------------------------------------------------- entity lists
ENTITY_TOOLS = [('record_entities', 'Save the entities mentioned in the text.'),
                ('extract_entities', 'Extract the named entities from a text.'),
                ('tag_text', 'Store the items found in a text, by kind.'),
                ('save_mentions', 'Record what the text mentions.')]


def _entity_row(text, found, absent, rng, source):
    """found: {type: [mentions]}. Fields are the found types plus some types the text lacks."""
    kinds = list(found) + rng.sample(absent, min(len(absent), rng.randint(0, 2)))
    rng.shuffle(kinds)
    name, description = rng.choice(ENTITY_TOOLS)
    properties = {re.sub(r'\W+', '_', k.lower()).strip('_'): {
        'type': 'array', 'items': {'type': 'string'}, 'description': f'Every {words(k)} in the text'} for k in kinds}
    arguments = {re.sub(r'\W+', '_', k.lower()).strip('_'): v for k, v in found.items()}
    return row(text, [tool(name, description, {'type': 'object', 'properties': properties})],
               [call(name, arguments)], source)


def _detokenize(tokens):
    text = ' '.join(tokens)
    text = re.sub(r" ([.,;:!?%)\]}]|'s|n't|'re|'ve|'ll|'d|'m)", r'\1', text)
    return re.sub(r'([(\[{$]) ', r'\1', text)


def few_nerd(limit=40000):
    path = hub.hf_file('DFKI-SLT/few-nerd', 'supervised/train-00000-of-00001.parquet')
    features = json.loads(pq.read_schema(path).metadata[b'huggingface'])['info']['features']
    coarse = features['ner_tags']['feature']['names']
    fine = [n.split('-', 1)[1] if '-' in n else n for n in features['fine_ner_tags']['feature']['names']]
    rng = random.Random(2)
    rows = hub.read_parquet(path)
    rng.shuffle(rows)
    for item in rows[:limit]:
        use_fine = rng.random() < 0.5
        names, tags = (fine, item['fine_ner_tags']) if use_fine else (coarse, item['ner_tags'])
        found, previous = {}, 0
        for token, tag in zip(item['tokens'], tags):
            if tag and names[tag] != 'other':
                mentions = found.setdefault(names[tag], [])
                if tag == previous:
                    mentions[-1] = _detokenize([mentions[-1], token])
                else:
                    mentions.append(token)
            previous = tag
        if not found or len(item['tokens']) < 6:
            continue
        found = {k: list(dict.fromkeys(v)) for k, v in found.items()}
        absent = [n for n in set(names[1:]) if n not in found and n != 'other']
        yield _entity_row(_detokenize(item['tokens']), found, absent, rng, 'few_nerd')


def pile_ner(max_chars=900):
    """Universal-NER Pile-NER: passages from many domains with open entity types."""
    rng = random.Random(3)
    for item in json.load(open(hub.hf_file('Universal-NER/Pile-NER-type', 'train.json'))):
        turns = item['conversations']
        text = turns[0]['value'].removeprefix('Text: ').strip()
        if len(text) > max_chars:
            continue
        found, absent = {}, []
        for question, answer in zip(turns[2::2], turns[3::2]):
            kind = re.fullmatch(r'What describes (.+) in the text\?', question['value'])
            if not kind or len(kind.group(1)) > 40:
                continue
            try:
                mentions = [m for m in json.loads(answer['value']) if isinstance(m, str) and m in text]
            except ValueError:
                continue
            (found.__setitem__(kind.group(1), list(dict.fromkeys(mentions))) if mentions
             else absent.append(kind.group(1)))
        if found:
            kinds = rng.sample(list(found), min(len(found), rng.randint(2, 5)))
            yield _entity_row(text, {k: found[k] for k in kinds}, absent, rng, 'pile_ner')


def generated(count=10000):
    yield from (row(*item, 'records') for item in records.generate(count, seed=4))


SOURCES = {'massive': massive, 'atis': atis, 'few_nerd': few_nerd, 'pile_ner': pile_ner, 'records': generated}
