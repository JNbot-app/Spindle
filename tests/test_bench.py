from spindle.bench.score import score, summarize
from spindle.convert import to_schema, tool


def case(kind, expected):
    return {'id': 'x', 'kind': kind, 'expected': expected}


def test_exact_calls_ignore_case_spacing_and_number_type():
    expected = [{'name': 'set_timer', 'arguments': {'minutes': 5, 'label': 'Pasta'}}]
    assert score(case('calls', expected), [{'name': 'set_timer', 'arguments': {'minutes': 5.0, 'label': ' pasta'}}])['correct']
    assert not score(case('calls', expected), [{'name': 'set_timer', 'arguments': {'minutes': 6, 'label': 'Pasta'}}])['correct']
    assert not score(case('calls', expected), [])['correct']


def test_ast_accepts_listed_values_and_optional_arguments():
    truth = [{'area': {'base': [10], 'unit': ['units', '']}}]
    assert score(case('ast', truth), [{'name': 'area', 'arguments': {'base': 10}}])['correct']
    assert score(case('ast', truth), [{'name': 'area', 'arguments': {'base': 10, 'unit': 'units'}}])['correct']
    assert not score(case('ast', truth), [{'name': 'area', 'arguments': {'base': 10, 'unit': 'cm'}}])['correct']
    assert not score(case('ast', truth), [{'name': 'area', 'arguments': {'unit': 'units'}}])['correct']


def test_field_f1_and_decision_rates():
    expected = [{'name': 'book', 'arguments': {'city': 'Ghent', 'nights': 2}}]
    records = [score(case('fields', expected), [{'name': 'book', 'arguments': {'city': 'ghent', 'nights': 3}}])]
    assert summarize(records)['score'] == 50.0
    records = [score(case('decision', False), []), score(case('decision', False), [{'name': 'a', 'arguments': {}}]),
               score(case('decision', True), [{'name': 'a', 'arguments': {}}])]
    summary = summarize(records)
    assert summary['decline_rate_when_not'] == 50.0 and summary['call_rate_when_needed'] == 100.0


def test_loose_schemas_become_json_schema():
    schema = to_schema({'type': 'dict', 'properties': {'n': {'type': 'float'}, 'tags': {'type': 'tuple', 'items': {'type': 'STRING'}}},
                        'required': ['n']})
    assert schema == {'type': 'object', 'required': ['n'], 'properties': {
        'n': {'type': 'number'}, 'tags': {'type': 'array', 'items': {'type': 'string'}}}}
    assert tool('t', None, None)['parameters'] == {'type': 'object', 'properties': {}}
