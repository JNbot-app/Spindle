"""Output filters: hold back calls the request does not support.

A small model sometimes calls a tool when it should not, or fills an argument with a value
the user never gave. These checks compare each call with the request text (query + system
prompt) only:

  * grounding: an argument value must come from the request. Numbers need a quantity in the
    request, enum values need the word (or a synonym), free text must appear in the request.
    Dates and times are exempt because the model may reformat them. An unsupported optional
    argument is dropped, a required one falls back to its schema default, and otherwise the
    whole call is held back.
  * negation: "don't open the door" holds back a call that would open it.
  * probability: calls below `min_call_probability` are held back.
  * duplicates: the same call twice is kept once.

Held-back calls go to `suppressed_calls`, so nothing is lost.
"""
import copy
import json
import re

from jsonschema import Draft202012Validator

_NUMBER = re.compile(r'(?<![\w.,])[-+]?\d+(?:[.,]\d+)*')
_NUMBER_WORD = re.compile(
    r'\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|'
    r'fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|'
    r'ninety|hundred|thousand|million|half|quarter|a couple|a few)\b', re.I)
_DATE_FACT = re.compile(r'\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?')

# Enum values that are usually said with another word ("turn the light up" -> "on").
SYNONYMS = {
    'open': ('open', 'unlock', 'unlatch'), 'close': ('close', 'shut', 'lock'),
    'on': ('on', 'enable', 'activate', 'start', 'brighten', 'dim'),
    'off': ('off', 'disable', 'deactivate', 'stop'),
    'forward': ('forward', 'forwards', 'ahead'), 'backward': ('backward', 'backwards', 'back'),
    'left': ('left', 'counterclockwise'), 'right': ('right', 'clockwise'),
}
# Tools whose output is a judgement about the text (not copied from it) skip grounding.
_JUDGEMENT = re.compile(r'classif|sentiment|detect|categor|analy[sz]', re.I)
_ACTION_VERBS = {'move', 'rotate', 'open', 'close', 'lock', 'unlock', 'send', 'delete', 'create',
                 'start', 'stop', 'play', 'pause', 'buy', 'sell', 'turn', 'remove', 'cancel'}
_NEGATION = r"\b(?:do\s+not|don['’]t|never|must\s+not)\s+(?:\w+\s+){0,2}"


def _words(text):
    return ' '.join(re.findall(r'\w+', str(text).replace('_', ' ').casefold()))


def mentioned(value, source):
    """True when `value` appears in `source` as whole words (case and punctuation ignored)."""
    value = _words(value)
    return bool(value) and f' {value} ' in f' {_words(source)} '


def _is_temporal(key, schema):
    return schema.get('format') in ('date', 'date-time', 'time') or bool(re.search(r'date|time', key))


class _Grounding:
    """Checks one call's arguments against the request; edits them in place."""

    def __init__(self, source):
        self.source = source
        self.has_quantity = bool(_NUMBER.search(_DATE_FACT.sub(' ', source)) or _NUMBER_WORD.search(source))
        self.missing = []

    def supported(self, key, value, schema):
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            return True
        if not isinstance(value, str):
            return self.has_quantity
        if schema.get('enum'):
            return any(mentioned(word, self.source) for word in SYNONYMS.get(value, (value,)))
        return _is_temporal(key, schema) or mentioned(value, self.source)

    def check(self, value, schema, path=''):
        if not isinstance(schema, dict):
            return
        choices = schema.get('anyOf', schema.get('oneOf'))
        if choices:
            schema = {**schema, **next((s for s in choices if Draft202012Validator(s).is_valid(value)), {})}
        for part in schema.get('allOf', []):
            self.check(value, part, path)
        if isinstance(value, dict):
            properties = schema.get('properties', {})
            for key in [k for k in value if isinstance(properties.get(k), dict)]:
                child, item = properties[key], value[key]
                child_path = f'{path}.{key}' if path else key
                if isinstance(item, (dict, list)):
                    self.check(item, child, child_path)
                elif not self.supported(key, item, child):
                    if key not in schema.get('required', []):
                        del value[key]
                    elif 'default' in child:
                        value[key] = copy.deepcopy(child['default'])
                    else:
                        self.missing.append(child_path)
        elif isinstance(value, list):
            prefix = schema.get('prefixItems', [])
            for i, item in enumerate(value):
                self.check(item, prefix[i] if i < len(prefix) else schema.get('items', {}), f'{path}[{i}]')


def _negated(call, query):
    """True when the query says not to do what this call does."""
    verb, *rest = call['name'].split('_')
    # The first word of a name like book_flight is its action; get_ and set_ say nothing.
    verbs = {verb} if verb in _ACTION_VERBS or (rest and verb not in ('get', 'set')) else set()
    for value in call['arguments'].values():
        if isinstance(value, str) and value in SYNONYMS:
            verbs.update(SYNONYMS[value])
    return any(re.search(_NEGATION + re.escape(v) + r'\b', query, re.I) for v in verbs)


def filter_calls(calls, tools, query, *, system='', call_probabilities=None, min_call_probability=0.0):
    """Split decoded calls into (kept, suppressed, reasons).

    `calls` and `tools` use the model's snake_case names; `reasons[i]` explains why the i-th
    suppressed call was held back.
    """
    schemas = {tool['name']: tool.get('parameters', {}) for tool in tools}
    source = query + '\n' + (system or '')
    kept, suppressed, reasons, seen = [], [], [], set()
    for index, call in enumerate(copy.deepcopy(calls)):
        reason = None
        probability = call_probabilities[index] if call_probabilities else 1.0
        if probability < min_call_probability:
            reason = f'probability {probability:.2f} is below {min_call_probability:.2f}'
        elif _negated(call, query):
            reason = 'the request says not to do this'
        elif not _JUDGEMENT.search(call['name']):
            grounding = _Grounding(source)
            grounding.check(call['arguments'], schemas[call['name']])
            if grounding.missing:
                reason = 'not in the request: ' + ', '.join(grounding.missing)
        if reason:
            suppressed.append(call)
            reasons.append(reason)
            continue
        key = json.dumps(call, sort_keys=True)
        if key not in seen:
            seen.add(key)
            kept.append(call)
    return kept, suppressed, reasons
