"""The native decoder (native/grammar.cpp) with a toy vocabulary: one token per byte plus the call markers."""
import numpy as np
import pytest

from spindle.inference.engine import Decoder, find_library

try:
    find_library()
except FileNotFoundError:
    pytest.skip('native engine not built', allow_module_level=True)

OPEN, CLOSE, EOS = 256, 257, 258
PIECES = [bytes([i]) for i in range(256)] + [b'<tool_call>', b'</tool_call>', b'']
LIGHT = [{'name': 'set_light', 'parameters': {
    'type': 'object', 'properties': {'state': {'type': 'string', 'enum': ['on', 'off']},
                                     'level': {'type': 'integer'}},
    'required': ['state'], 'additionalProperties': False}}]


def decode(target, *, require_call=False, query=''):
    """Greedy decoding where the toy model wants the next byte of `target`, then the end markers."""
    decoder = Decoder(None, PIECES, [0] * 258 + [1], OPEN, CLOSE, None, {EOS})
    decoder.set_tools(LIGHT, {'on': ord('o'), 'off': ord('o')})
    decoder.start(query, require_call=require_call)
    written = b''
    base = np.zeros(len(PIECES), np.float32)  # when the wanted byte is not allowed: close, quote, letters
    base[[ord('}'), ord(']')]] = 3
    base[ord('"')] = 2
    base[ord('a'):ord('z') + 1] = 1
    base[CLOSE] = base[EOS] = 5
    for _ in range(400):
        logits = base.copy()
        if len(written) < len(target):
            logits[target[len(written)]] = 10
        token = decoder.step(logits)
        if token < 256:
            written += bytes([token])
        if decoder.done:
            break
    return decoder.result()


def test_valid_call_is_kept():
    result = decode(b'[{"name":"set_light","arguments":{"state":"off","level":3}}]')
    assert result['complete']
    assert result['calls'] == [{'name': 'set_light', 'arguments': {'state': 'off', 'level': 3}}]
    assert 0 < result['probabilities'][0] <= 1


def test_grammar_blocks_invalid_values():
    # "of" is not an enum label and 3.5 is not an integer: the grammar forces valid continuations.
    result = decode(b'[{"name":"set_light","arguments":{"state":"ofx","level":3.5}}]')
    assert result['complete']
    call = result['calls'][0]
    assert call['name'] == 'set_light' and call['arguments']['state'] in ('on', 'off')
    assert isinstance(call['arguments'].get('level', 0), int)


def test_empty_answer_and_require_call():
    assert decode(b'[]')['calls'] == []
    assert decode(b'[]', require_call=True)['calls'][0]['name'] == 'set_light'
