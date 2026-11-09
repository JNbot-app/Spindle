import json

from spindle.prompt import render_example
from spindle.training.data import argument_spans
from spindle.training.spans import copied_values

ROW = {'query': 'Set a 10 minute timer called pasta', 'tools': [],
       'answers': [{'name': 'set_timer', 'arguments': {'minutes': 10, 'label': 'pasta', 'loud': True}}]}


def test_argument_spans_cover_values_only():
    _, target = render_example(ROW)
    values = [target[a:b] for a, b in argument_spans(target)]
    assert values == ['10', 'pasta', 'true']


def test_copied_values_match_whole_numbers():
    found = {name: ROW['query'][a:b] for _, name, a, b in copied_values(ROW)}
    assert found == {'minutes': '10', 'label': 'pasta'}
    row = dict(ROW, query='Set a 100 minute timer called pasta')
    assert 'minutes' not in {name for _, name, _, _ in copied_values(row)}


def test_target_format():
    _, target = render_example(ROW)
    assert target.startswith('<tool_call>[') and target.endswith('</tool_call><|im_end|>')
    assert json.loads(target[len('<tool_call>'):-len('</tool_call><|im_end|>')]) == ROW['answers']
