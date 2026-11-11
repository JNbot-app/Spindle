from spindle.inference.filters import filter_calls

TOOLS = [
    {'name': 'set_timer', 'parameters': {'type': 'object', 'properties': {
        'minutes': {'type': 'integer'}, 'label': {'type': 'string'}}, 'required': ['minutes']}},
    {'name': 'set_light', 'parameters': {'type': 'object', 'properties': {
        'room': {'type': 'string'}, 'state': {'type': 'string', 'enum': ['on', 'off']}},
        'required': ['room', 'state']}},
    {'name': 'open_door', 'parameters': {'type': 'object', 'properties': {}}},
]


def run(calls, query, **options):
    return filter_calls(calls, TOOLS, query, **options)


def test_grounded_call_is_kept():
    call = {'name': 'set_timer', 'arguments': {'minutes': 5}}
    assert run([call], 'Set a timer for 5 minutes') == ([call], [], [])


def test_number_without_quantity_is_suppressed():
    kept, suppressed, reasons = run([{'name': 'set_timer', 'arguments': {'minutes': 5}}], 'Tell me a joke')
    assert not kept and len(suppressed) == 1 and 'minutes' in reasons[0]


def test_unsupported_optional_argument_is_dropped():
    kept, _, _ = run([{'name': 'set_timer', 'arguments': {'minutes': 10, 'label': 'pasta'}}],
                     'Start a ten minute timer')
    assert kept == [{'name': 'set_timer', 'arguments': {'minutes': 10}}]


def test_enum_synonym_counts_as_grounded():
    call = {'name': 'set_light', 'arguments': {'room': 'kitchen', 'state': 'off'}}
    assert run([call], 'Please disable the kitchen light')[0] == [call]


def test_negation_and_duplicates():
    door = {'name': 'open_door', 'arguments': {}}
    assert run([door], "Don't open the door")[1] == [door]
    assert run([door, door], 'Open the door')[0] == [door]


def test_probability_floor():
    call = {'name': 'set_timer', 'arguments': {'minutes': 5}}
    kept, suppressed, _ = run([call], 'Set a timer for 5 minutes', call_probabilities=[0.5],
                              min_call_probability=0.9)
    assert not kept and suppressed == [call]
