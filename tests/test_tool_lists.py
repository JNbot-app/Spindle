import random

from spindle.training import tool_lists


def tool(name):
    return {'name': name, 'description': name, 'parameters': {'type': 'object', 'properties': {}}}


ROWS = [
    {'query': 'Weather in Lagos?', 'tools': [tool('get_weather'), tool('set_timer')],
     'answers': [{'name': 'get_weather', 'arguments': {'city': 'Lagos'}}]},
    {'query': 'Timer for 5 minutes', 'tools': [tool('set_timer')],
     'answers': [{'name': 'set_timer', 'arguments': {'minutes': 5}}]},
    {'query': 'Text Ann hi', 'tools': [tool('send_message'), tool('set_light')],
     'answers': [{'name': 'send_message', 'arguments': {'to': 'Ann', 'text': 'hi'}}]},
    {'query': 'Lights off', 'tools': [tool('set_light'), tool('send_email'), tool('get_forecast')],
     'answers': [{'name': 'set_light', 'arguments': {'on': False}}]},
    {'query': 'Thanks!', 'tools': [tool('set_light')], 'answers': []},
]


def names(row):
    return {t['name'] for t in row['tools']}


def test_varied_keeps_answer_and_needed_tool():
    pool = tool_lists.tool_pool(ROWS)
    for seed in range(20):
        row = tool_lists.varied(ROWS[0], pool, random.Random(seed))
        assert row['answers'] == ROWS[0]['answers'] and 'get_weather' in names(row)
        assert 'get_forecast' not in names(row)  # could do the same job


def test_absent_removes_needed_tool_and_look_alikes():
    pool = tool_lists.tool_pool(ROWS)
    for seed in range(20):
        row = tool_lists.absent(ROWS[2], pool, random.Random(seed))
        assert row['answers'] == [] and row['query'] == ROWS[2]['query']
        assert not names(row) & {'send_message', 'send_email'}


def test_augment_adds_absent_rows_and_respects_fits():
    out = tool_lists.augment(ROWS, 1.0, 0.4, fits=lambda row: len(row['tools']) <= 3)
    assert len(out) == len(ROWS) + 2
    assert all(len(row['tools']) <= 3 for row in out if row not in ROWS)
    assert out[4] is ROWS[4]  # no-call rows are left alone
