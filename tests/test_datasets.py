import itertools

from datetime import datetime

from spindle.datasets import dates, mix, records
from spindle.datasets.tool_calling import _parse_call, _split_calls


def test_generated_records_only_answer_stated_fields():
    for text, tools, answers in itertools.islice(records.generate(200, seed=1), 200):
        properties = tools[0]['parameters']['properties']
        arguments = answers[0]['arguments']
        assert arguments and set(arguments) <= set(properties)
        for value in arguments.values():
            if isinstance(value, str) and 'enum' not in str(properties):
                assert value in text


def test_allocate_fills_shortfalls_from_other_sources():
    take = mix.allocate(100, {'a': 0.5, 'b': 0.3, 'c': 0.2}, {'a': 10, 'b': 1000, 'c': 1000})
    assert sum(take.values()) == 100 and take['a'] == 10 and take['b'] > take['c']
    assert sum(mix.allocate(100, {'a': 1.0}, {'a': 30}).values()) == 30


def test_python_style_calls_are_parsed():
    parts = _split_calls('[Get Weather(city="Lagos, NG", days=2), ping()]')
    assert parts == ['Get Weather(city="Lagos, NG", days=2)', 'ping()']
    assert _parse_call(parts[0]) == {'name': 'Get Weather', 'arguments': {'city': 'Lagos, NG', 'days': 2}}
    assert _split_calls('Sorry, I cannot do that.') is None


def test_date_rows_resolve_relative_days():
    import random
    now = datetime(2024, 6, 6, 9, 58)  # a Thursday
    for _ in range(300):
        day, phrase = dates._day(now, random.Random(_))
        if phrase == 'tomorrow':
            assert day.date() == datetime(2024, 6, 7).date()
        if phrase.endswith('Tuesday') and not phrase.startswith('on Tuesday,'):
            assert day.date() == datetime(2024, 6, 11).date()
        if phrase.endswith('Thursday') and not phrase.startswith('on Thursday,'):
            assert day.date() == datetime(2024, 6, 13).date()  # never today
    for row in dates.generate(200, seed=1):
        assert row['system'] and row['answers'][0]['name'] in {t['name'] for t in row['tools']}


def test_reasoning_note_and_target():
    from spindle.prompt import render_example
    from spindle.reasoning import note
    row = {'query': 'Show Piazza Navona in Rome', 'tools': [],
           'answers': [{'name': 'show_map', 'arguments': {'query': 'Piazza Navona in Rome', 'zoom': 3}}]}
    assert note(row) == "show_map: query 'Piazza Navona in Rome' from query; zoom '3'."
    assert note({'query': 'hi', 'answers': []}).startswith('No')
    target = render_example({**row, 'reasoning': note(row)})[1]
    assert target.startswith('<think>\nshow_map: query') and '\n</think>\n<tool_call>[{' in target
    assert render_example(row)[1].startswith('<tool_call>')


def test_combined_rows_join_two_requests():
    from spindle.datasets import combined
    def single(name, query):
        return {'query': query, 'tools': [{'name': name, 'description': '', 'parameters': {}}],
                'answers': [{'name': name, 'arguments': {}}], 'task': 'tools'}
    rows = combined.generate({'a': [single('light_on', 'Turn on the light'), single('play', 'Play jazz.')]}, 4, seed=1)
    assert rows and all(len(r['answers']) == 2 and len(r['tools']) == 2 for r in rows)
    assert all({'Turn on the light', 'jazz'} <= {w for w in ('Turn on the light', 'jazz') if w.lower() in r['query'].lower()}
               for r in rows)
