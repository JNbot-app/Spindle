"""Classification rows: one tool with an enum field (task 'classify'), and intent routing, where
every intent is a tool without arguments (task 'tools', or 'no_call' when no listed intent fits).

Tool names, field names, label words and label order vary from row to row, so the answer can
only come from reading the schema. Intent sets (Banking77, CLINC150) also give long enum lists of
similar labels (up to 20), so the model learns to pick among many close choices.
"""
import json
import random
import re

import pyarrow.parquet as pq

from spindle import hub
from spindle.convert import call, tool


def _row(query, tools, answers, task, source):
    return {'query': query.strip(), 'tools': tools, 'answers': answers, 'task': task, 'source': source}


def _label_names(path, column):
    features = json.loads(pq.read_schema(path).metadata[b'huggingface'])['info']['features']
    return features[column]['names']


def _long_enum_rows(examples, names, source, tool_names, field_names, description, seed, most=16):
    """examples: (text, label index). The right label among 3 to `most - 1` other labels of the set."""
    rng = random.Random(seed)
    for text, label in examples:
        shown = [names[label], *rng.sample([n for n in names if n != names[label]], rng.randint(3, most - 1))]
        rng.shuffle(shown)
        name, field = rng.choice(tool_names), rng.choice(field_names)
        parameters = {'type': 'object', 'properties': {field: {'type': 'string', 'enum': shown}}, 'required': [field]}
        yield _row(text, [tool(name, description, parameters)], [call(name, {field: names[label]})], 'classify', source)


def _enum_rows(examples, source, tool_names, field_names, descriptions, label_sets, seed, extra_labels=()):
    """examples: (text, label index). label_sets: alternative wordings of the same label list."""
    rng = random.Random(seed)
    for text, label in examples:
        labels = rng.choice(label_sets)
        answer = labels[label]
        shown = list(labels)
        if extra_labels and rng.random() < 0.3:
            shown.append(rng.choice([e for e in extra_labels if e not in shown]))
        rng.shuffle(shown)
        name, field = rng.choice(tool_names), rng.choice(field_names)
        spec = {'type': 'string', 'enum': shown}
        if rng.random() < 0.5:
            spec['description'] = rng.choice(descriptions)
        parameters = {'type': 'object', 'properties': {field: spec}, 'required': [field]}
        yield _row(text, [tool(name, rng.choice(descriptions), parameters)], [call(name, {field: answer})],
                   'classify', source)


SENTIMENT_TOOLS = ['classify_sentiment', 'sentiment', 'rate_text', 'label_review', 'analyze_sentiment', 'tag_opinion']
SENTIMENT_FIELDS = ['sentiment', 'label', 'polarity', 'tone', 'rating']
SENTIMENT_TEXT = ['Classify the sentiment of a text.', 'Decide how the writer feels about the subject.',
                  'Overall sentiment of the text', 'Label the opinion expressed in a message.']


def sst2(limit=12000):
    rows = hub.hf_parquet('stanfordnlp/sst2', 'data/train-00000-of-00001.parquet')
    rows = [(r['sentence'].strip(), r['label']) for r in rows if len(r['sentence'].split()) >= 5]
    random.Random(10).shuffle(rows)
    yield from _enum_rows(rows[:limit], 'sst2', SENTIMENT_TOOLS, SENTIMENT_FIELDS, SENTIMENT_TEXT,
                          [['negative', 'positive'], ['bad', 'good'], ['neg', 'pos'], ['unfavorable', 'favorable'],
                           ['dislike', 'like']], 10, extra_labels=['neutral', 'mixed'])


def tweet_sentiment(limit=12000):
    rows = hub.hf_parquet('cardiffnlp/tweet_eval', 'sentiment/train-00000-of-00001.parquet')
    rows = [(r['text'].strip().strip('"'), r['label']) for r in rows]
    random.Random(11).shuffle(rows)
    yield from _enum_rows(rows[:limit], 'tweet_sentiment', SENTIMENT_TOOLS, SENTIMENT_FIELDS, SENTIMENT_TEXT,
                          [['negative', 'neutral', 'positive'], ['bad', 'neutral', 'good'], ['neg', 'neu', 'pos'],
                           ['unhappy', 'indifferent', 'happy']], 11)


def emotion(limit=8000):
    rows = hub.hf_parquet('dair-ai/emotion', 'split/train-00000-of-00001.parquet')
    rows = [(r['text'], r['label']) for r in rows]
    random.Random(12).shuffle(rows)
    yield from _enum_rows(rows[:limit], 'emotion', ['classify_emotion', 'detect_emotion', 'tag_feeling', 'emotion'],
                          ['emotion', 'feeling', 'label', 'mood'],
                          ['Name the main emotion in a text.', 'Which emotion the writer expresses'],
                          [['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'],
                           ['sad', 'happy', 'loving', 'angry', 'afraid', 'surprised']], 12,
                          extra_labels=['disgust', 'boredom'])


def ag_news(limit=8000):
    rows = hub.hf_parquet('fancyzhx/ag_news', 'data/train-00000-of-00001.parquet')
    rows = [(r['text'].replace('\\', ' '), r['label']) for r in rows if len(r['text']) < 400]
    random.Random(13).shuffle(rows)
    yield from _enum_rows(rows[:limit], 'ag_news', ['classify_topic', 'tag_article', 'route_news', 'news_section'],
                          ['topic', 'category', 'section', 'label'],
                          ['Pick the topic of a news item.', 'News section the text belongs to'],
                          [['world', 'sports', 'business', 'science and technology'],
                           ['international', 'sport', 'finance', 'tech'], ['World', 'Sports', 'Business', 'Sci/Tech']],
                          13, extra_labels=['health', 'entertainment', 'weather'])


# ---------------------------------------------------------------- intent routing
def _routing(examples, names, source, seed, descriptions=None, none_label=None):
    """Each intent is a tool without arguments. Rows whose intent is `none_label`, or whose intent
    is left out of the list on purpose, have no call as the answer."""
    rng = random.Random(seed)
    intents = [n for n in names if n != none_label]

    def spec(name):
        text = (descriptions or {}).get(name) or f'The user wants: {re.sub("_+", " ", name)}.'
        return tool(name, text, {'type': 'object', 'properties': {}})

    for text, label in examples:
        name = names[label]
        outside = name == none_label
        others = rng.sample([i for i in intents if i != name], rng.randint(2, 7))
        if outside or rng.random() < 0.1:
            listed, answers = others, []
        else:
            listed, answers = [name, *others], [call(name, {})]
        rng.shuffle(listed)
        yield _row(text, [spec(i) for i in listed], answers, 'tools' if answers else 'no_call', source)


def clinc(limit=16000):
    """CLINC150: 150 intents plus out-of-scope requests (no call)."""
    path = hub.hf_file('clinc/clinc_oos', 'plus/train-00000-of-00001.parquet')
    names = _label_names(path, 'intent')
    rows = [(r['text'], r['intent']) for r in hub.read_parquet(path)]
    random.Random(14).shuffle(rows)
    outside = [r for r in rows if names[r[1]] == 'oos']
    inside = [r for r in rows if names[r[1]] != 'oos'][:limit - len(outside)]
    rows = inside + outside
    random.Random(15).shuffle(rows)
    yield from _routing(rows, names, 'clinc', 14, none_label='oos')


def banking(limit=8000):
    path = hub.hf_file('PolyAI/banking77', 'data/train-00000-of-00001.parquet')
    names = _label_names(path, 'label')
    rows = [(r['text'], r['label']) for r in hub.read_parquet(path)]
    random.Random(16).shuffle(rows)
    half = min(limit, len(rows)) // 2
    yield from _routing(rows[:half], names, 'banking77', 16)
    # The other half as an enum: the right label among up to 15 others.
    yield from _long_enum_rows(rows[half:2 * half], names, 'banking77', ['classify_request', 'route_ticket', 'tag_question'],
                               ['intent', 'topic', 'label'], 'Pick what a bank customer is asking about.', 17)


def clinc_labels(limit=8000):
    """CLINC150 as an enum: the right intent (in words) among up to 19 others."""
    path = hub.hf_file('clinc/clinc_oos', 'plus/train-00000-of-00001.parquet')
    all_names = _label_names(path, 'intent')
    index = {i: k for k, i in enumerate(i for i, n in enumerate(all_names) if n != 'oos')}  # out-of-scope is no label
    names = [n.replace('_', ' ') for n in all_names if n != 'oos']
    rows = [(r['text'], index[r['intent']]) for r in hub.read_parquet(path) if r['intent'] in index]
    random.Random(18).shuffle(rows)
    yield from _long_enum_rows(rows[:limit], names, 'clinc_labels',
                               ['detect_intent', 'classify_request', 'route_message', 'intent'],
                               ['intent', 'request_type', 'category', 'label'], 'What the user is asking for.', 18, most=20)


SOURCES = {'sst2': sst2, 'tweet_sentiment': tweet_sentiment, 'emotion': emotion, 'ag_news': ag_news,
           'clinc': clinc, 'banking77': banking, 'clinc_labels': clinc_labels}
