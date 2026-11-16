"""Grammar-constrained decoding: only tokens that keep the output a valid tool-call list.

The output must be a JSON array of {"name": ..., "arguments": ...} objects, where every call
matches one tool's JSON schema. JsonState follows the JSON byte by byte; the sampler picks the
most likely token whose bytes keep that state valid.
"""
import copy
from functools import lru_cache
import json
import re

import numpy as np
from jsonschema import Draft202012Validator


class DecodingError(RuntimeError):
    pass


def expand_refs(value, root=None, seen=()):
    """Inline local "$ref" pointers (non-recursive schemas only)."""
    root = value if root is None else root
    if isinstance(value, list):
        return [expand_refs(v, root, seen) for v in value]
    if not isinstance(value, dict):
        return value
    if '$ref' in value:
        ref = value['$ref']
        if not isinstance(ref, str) or not ref.startswith('#/') or ref in seen:
            raise ValueError('Only non-recursive local JSON schema references are supported')
        target = root
        try:
            for part in ref[2:].split('/'):
                target = target[part.replace('~1', '/').replace('~0', '~')]
        except (KeyError, TypeError) as error:
            raise ValueError('Unresolved local schema reference') from error
        return {'allOf': [expand_refs(target, root, (*seen, ref)),
                          expand_refs({k: v for k, v in value.items() if k != '$ref'}, root, seen)]}
    return {key: expand_refs(item, root, seen) for key, item in value.items()}


def default_additional_properties(schema):
    """Objects allow no extra keys unless the schema says otherwise."""
    if isinstance(schema, list):
        return [default_additional_properties(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    result = dict(schema)
    if isinstance(result.get('properties'), dict):
        result['properties'] = {k: default_additional_properties(v) for k, v in result['properties'].items()}
    if 'items' in result:
        result['items'] = default_additional_properties(result['items'])
    for key in ('prefixItems', 'oneOf', 'anyOf', 'allOf'):
        if isinstance(result.get(key), list):
            result[key] = [default_additional_properties(v) for v in result[key]]
    for key in ('$defs', 'definitions'):
        if isinstance(result.get(key), dict):
            result[key] = {k: default_additional_properties(v) for k, v in result[key].items()}
    types = result.get('type')
    if ('properties' in result or types == 'object' or (isinstance(types, list) and 'object' in types)) \
            and 'additionalProperties' not in result:
        result['additionalProperties'] = False
    if isinstance(result.get('additionalProperties'), dict):
        result['additionalProperties'] = default_additional_properties(result['additionalProperties'])
    return result


class InvalidPrefix(ValueError):
    pass


def variants(schema):
    if schema is False:
        return []
    if schema is True:
        return [{}]
    branches = schema.get('oneOf', schema.get('anyOf', [schema]))
    result = []
    for branch in branches:
        projected = dict(branch)
        for part in branch.get('allOf', []):
            options = variants(part)
            if len(options) != 1:
                # Container completion still checks the original full schema.
                continue
            part = options[0]
            for key in ('type', 'items', 'prefixItems', 'enum', 'const'):
                if key not in projected and key in part:
                    projected[key] = part[key]
            if 'properties' in part:
                properties = dict(projected.get('properties', {}))
                for key, value in part['properties'].items():
                    properties[key] = {'allOf': [properties[key], value]} if key in properties else value
                projected['properties'] = properties
            if 'additionalProperties' in part:
                projected['additionalProperties'] = part['additionalProperties']
            projected['required'] = list(set(projected.get('required', [])) | set(part.get('required', [])))
        if ('properties' in projected or projected.get('type') == 'object') and 'additionalProperties' not in projected:
            projected['additionalProperties'] = False
        result.append(projected)
    return result


def permits(schema, kind):
    for item in variants(schema):
        types = item.get('type')
        allowed = [types] if isinstance(types, str) else types
        if types is None or kind in allowed or (kind == 'number' and 'integer' in allowed):
            if all(permits(part, kind) for part in item.get('allOf', [])):
                return True
    return False


@lru_cache(maxsize=512)
def string_encodings(value):
    """Legal JSON spellings of each character, including Unicode escapes."""
    escapes = {'"': b'\\"', '\\': b'\\\\', '/': b'\\/', '\b': b'\\b',
               '\f': b'\\f', '\n': b'\\n', '\r': b'\\r', '\t': b'\\t'}
    result = []
    for character in value:
        units = character.encode('utf-16-be', 'surrogatepass').hex()
        unicode_escape = ''.join('\\u' + units[i:i + 4] for i in range(0, len(units), 4)).encode()
        options = [(unicode_escape, True)]
        if character in escapes:
            options.append((escapes[character], False))
        if ord(character) >= 32 and character not in ('"', '\\') and not 0xd800 <= ord(character) <= 0xdfff:
            options.append((character.encode('utf-8'), False))
        result.append(tuple(options))
    return tuple(result)


def string_prefix_matches(raw, choices):
    """Keep prefixes only when an allowed string can still complete them."""
    if b'\\' not in raw:
        return any(value.encode('utf-8', 'surrogatepass').startswith(raw) for value in choices)
    for choice in choices:
        positions = {0}
        for encodings in string_encodings(choice):
            following = set()
            for position in positions:
                suffix = raw[position:]
                for spelling, unicode_escape in encodings:
                    fragment = suffix[:len(spelling)]
                    if unicode_escape:
                        fragment = fragment.lower()
                    if spelling.startswith(fragment):
                        if len(suffix) <= len(spelling):
                            return True
                        following.add(position + len(spelling))
            positions = following
            if not positions:
                break
        if len(raw) in positions:
            return True
    return False


class JsonState:
    def __init__(self, schema):
        self.schema = default_additional_properties(schema)
        self.stack = []
        self.mode = None
        self.raw = b''
        self.escape = False
        self.unicode_remaining = 0
        self.scalar_schema = None
        self.string_key = False
        self.done = False
        self.value = None
        self.last_was_space = False

    def clone(self):
        other = copy.copy(self)
        other.stack = [dict(frame, value=copy.deepcopy(frame['value'])) for frame in self.stack]
        other.last_was_space = getattr(self, 'last_was_space', False)
        return other

    @staticmethod
    def _is_int_only(schema):
        return permits(schema, 'integer') and not any(
            item.get('type') == 'number' or (isinstance(item.get('type'), list) and 'number' in item['type'])
            for item in variants(schema)
        )

    @staticmethod
    def check(value, schema):
        schema = default_additional_properties(schema)
        if not Draft202012Validator(schema).is_valid(value):
            raise InvalidPrefix('Value violates schema')

    def effective(self, frame):
        options = variants(frame['schema'])
        if frame['kind'] == 'object' and 'name' in frame['value']:
            selected = [s for s in options if s.get('properties', {}).get('name', {}).get('const', frame['value']['name']) == frame['value']['name']]
            if selected:
                options = selected
        return options

    def available_keys(self, frame):
        options = self.effective(frame)
        if any(option.get('additionalProperties', False) is not False for option in options):
            return None
        return {key for option in options for key in option.get('properties', {}) if key not in frame['value']}

    def expected(self):
        if not self.stack:
            return self.schema
        frame = self.stack[-1]
        options = self.effective(frame)
        candidates = []
        for item in options:
            if frame['kind'] == 'array':
                position = len(frame['value'])
                candidates.append(item.get('prefixItems', [])[position] if position < len(item.get('prefixItems', [])) else item.get('items', {}))
            else:
                candidates.append(item.get('properties', {}).get(frame['key'], item.get('additionalProperties', False)))
        candidates = [s for s in candidates if s is not False]
        if not candidates:
            raise InvalidPrefix('No allowed value')
        return candidates[0] if len(candidates) == 1 else {'anyOf': candidates}

    def emit(self, value, schema):
        self.check(value, schema)
        if not self.stack:
            self.value, self.done = value, True
            return
        frame = self.stack[-1]
        if frame['kind'] == 'array':
            frame['value'].append(value)
        else:
            frame['value'][frame['key']] = value
        frame['stage'] = 'delimiter'

    def close(self):
        frame = self.stack.pop()
        self.emit(frame['value'], frame['schema'])

    def byte(self, byte):
        if self.mode == 'string':
            self.last_was_space = False
            if byte < 32:
                raise InvalidPrefix('Control byte in string')
            if byte == 34 and not self.escape:
                try:
                    value = json.loads(b'"' + self.raw + b'"')
                except (ValueError, UnicodeError) as error:
                    raise InvalidPrefix('Invalid JSON string') from error
                self.mode = None
                if self.string_key:
                    frame = self.stack[-1]
                    if value in frame['value']:
                        raise InvalidPrefix('Duplicate key')
                    frame['key'], frame['stage'] = value, 'colon'
                    self.expected()
                else:
                    self.emit(value, self.scalar_schema)
                return
            if self.unicode_remaining:
                if byte not in b'0123456789abcdefABCDEF':
                    raise InvalidPrefix('Invalid unicode escape')
                self.unicode_remaining -= 1
            elif self.escape:
                if byte not in b'"\\/bfnrtu':
                    raise InvalidPrefix('Invalid escape')
                self.unicode_remaining = 4 if byte == 117 else 0
                self.escape = False
            else:
                self.escape = byte == 92
            self.raw += bytes([byte])
            if self.string_key:
                frame = self.stack[-1]
                keys = self.available_keys(frame)
                if keys is not None and not string_prefix_matches(self.raw, keys):
                    raise InvalidPrefix('Key cannot match an allowed property')
            if b'\\' not in self.raw:
                try:
                    self.raw.decode('utf-8')
                except UnicodeDecodeError as error:
                    if error.reason != 'unexpected end of data':
                        raise InvalidPrefix('Invalid UTF-8') from error
            if not self.string_key:
                choices = []
                unconstrained = False
                for option in variants(self.scalar_schema):
                    values = option.get('enum', [option['const']] if 'const' in option else None)
                    if values is None:
                        unconstrained = True
                    else:
                        choices.extend(v for v in values if isinstance(v, str))
                if not unconstrained and not string_prefix_matches(self.raw, choices):
                    raise InvalidPrefix('String cannot match enum')
            return
        if self.mode == 'number':
            self.last_was_space = False
            int_only = self._is_int_only(self.scalar_schema)
            allowed = b'0123456789' if int_only else b'0123456789.eE+-'
            if byte in allowed:
                self.raw += bytes([byte])
                pattern = rb'-?(?:0|[1-9][0-9]*)' if int_only else rb'-?(?:0|[1-9][0-9]*)(?:(?:\.[0-9]+)?(?:[eE][+-]?[0-9]*)?|\.[0-9]*)'
                if self.raw != b'-' and not re.fullmatch(pattern, self.raw):
                    raise InvalidPrefix('Invalid number prefix')
                return
            try:
                value = int(self.raw) if int_only else json.loads(self.raw)
            except ValueError as error:
                raise InvalidPrefix('Invalid number') from error
            self.mode = None
            self.emit(value, self.scalar_schema)
            self.byte(byte)
            return
        if self.mode == 'literal':
            self.last_was_space = False
            self.raw += bytes([byte])
            targets = [v for v in (b'true', b'false', b'null') if v.startswith(self.raw)]
            if not targets:
                raise InvalidPrefix('Invalid literal')
            if self.raw in targets:
                self.mode = None
                self.emit(json.loads(self.raw), self.scalar_schema)
            return
        if byte in b' \r\n\t':
            if byte != 32 or self.last_was_space:
                raise InvalidPrefix('Unnecessary whitespace')
            self.last_was_space = True
            return
        self.last_was_space = False
        if self.done:
            raise InvalidPrefix('Trailing data')
        if self.stack:
            frame = self.stack[-1]
            stage = frame['stage']
            if stage == 'colon':
                if byte != 58:
                    raise InvalidPrefix('Expected colon')
                frame['stage'] = 'value'
                return
            if stage == 'delimiter':
                if byte == (125 if frame['kind'] == 'object' else 93):
                    self.close()
                    return
                if byte != 44:
                    raise InvalidPrefix('Expected comma')
                if frame['kind'] == 'object' and self.available_keys(frame) == set():
                    raise InvalidPrefix('No further object properties are allowed')
                frame['stage'] = 'key' if frame['kind'] == 'object' else 'value'
                return
            if stage in ('key', 'key_or_end'):
                if byte == 125 and stage == 'key_or_end':
                    self.close()
                elif byte == 34:
                    if self.available_keys(frame) == set():
                        raise InvalidPrefix('No object properties are allowed')
                    self.mode, self.raw, self.escape, self.string_key = 'string', b'', False, True
                else:
                    raise InvalidPrefix('Expected object key')
                return
            if stage == 'value_or_end' and byte == 93:
                self.close()
                return
        schema = self.expected()
        if byte in (123, 91):
            kind = 'object' if byte == 123 else 'array'
            if not permits(schema, kind):
                raise InvalidPrefix('Wrong container type')
            self.stack.append({'schema': schema, 'kind': kind, 'value': {} if kind == 'object' else [],
                               'stage': 'key_or_end' if kind == 'object' else 'value_or_end', 'key': None})
        elif byte == 34:
            if not permits(schema, 'string'):
                raise InvalidPrefix('Wrong string type')
            self.mode, self.raw, self.escape, self.string_key, self.scalar_schema = 'string', b'', False, False, schema
        elif byte in b'-0123456789':
            if not permits(schema, 'number'):
                raise InvalidPrefix('Wrong number type')
            self.mode, self.raw, self.scalar_schema = 'number', bytes([byte]), schema
        elif byte in b'tfn':
            kind = 'null' if byte == 110 else 'boolean'
            if not permits(schema, kind):
                raise InvalidPrefix('Wrong literal type')
            self.mode, self.raw, self.scalar_schema = 'literal', bytes([byte]), schema
        else:
            raise InvalidPrefix('Expected JSON value')

    def feed(self, data):
        for byte in data:
            self.byte(byte)


def token_bytes(tokenizer):
    """The bytes each token stands for. Built once per tokenizer."""
    pieces = getattr(tokenizer, '_token_bytes', None)
    if pieces is None:
        pieces = tokenizer._token_bytes = [
            bytes([int(p[3:5], 16)]) if kind == 4 else p.replace('\u2581', ' ').encode('utf-8')
            for p, kind in zip(tokenizer.pieces, tokenizer.types)]
    return pieces


class IncrementalSampler:
    """Tracks the generated tool-call JSON and picks the best grammar-valid token."""

    def __init__(self, tools, tokenizer, require_call=False, check=True):
        """`check=False` skips the schema validation (for tools that were validated before)."""
        self.require_call = require_call  # the answer may not be the empty list
        alternatives = [{'type': 'object',
                         'properties': {'name': {'const': tool['name']}, 'arguments': expand_refs(tool['parameters'])},
                         'required': ['name', 'arguments'], 'additionalProperties': False} for tool in tools]
        schema = {'type': 'array', 'items': {'oneOf': alternatives} if alternatives else False}
        if check:
            Draft202012Validator.check_schema(schema)
        self.json = JsonState(schema)
        self.mode = 'before'  # before -> json -> closed -> done
        self.tokenizer = tokenizer
        self.open_id = tokenizer.p2id['<tool_call>']
        self.close_id = tokenizer.p2id['</tool_call>']
        self.stop_ids = {tokenizer.eos_id, tokenizer.p2id['<|im_end|>']}
        self.pieces = token_bytes(tokenizer)
        self.transitions = {}

    def transition(self, token):
        """(next JSON state, next mode) after `token`, or None if the token is not allowed."""
        if token in self.transitions:
            return self.transitions[token]
        state, mode = self.json, self.mode
        special = self.tokenizer.types[token] in (1, 2)
        if token in self.stop_ids:
            result = (state, 'done') if mode == 'closed' else None
        elif mode == 'before':
            result = ((state, 'json') if token == self.open_id else
                      (state, mode) if token != self.close_id and not special else None)
        elif mode == 'closed':
            result = None
        elif token == self.close_id:
            result = (state, 'closed') if state.done else None
        elif token == self.open_id or special:
            result = None
        else:
            state = state.clone()
            try:
                state.feed(self.pieces[token])
                result = None if self.require_call and state.done and not state.value else (state, mode)
            except (InvalidPrefix, ValueError, UnicodeError):
                result = None
        self.transitions[token] = result
        return result

    def select_token(self, logits):
        """The most likely allowed token: the best one, then the top 32, then the whole vocabulary."""
        values = np.asarray(logits)
        if values.ndim != 1 or len(values) != len(self.pieces) or not np.isfinite(values).all():
            raise DecodingError('Invalid decoder logits')
        best = int(values.argmax())  # the first of equal values, like the stable sorts below
        if self.transition(best) is not None:
            return best
        top = np.argpartition(-values, 31)[:32]
        for token in top[np.argsort(-values[top], kind='stable')]:
            if self.transition(int(token)) is not None:
                return int(token)
        for token in np.argsort(-values, kind='stable'):  # rare: sort everything only when needed
            if self.transition(int(token)) is not None:
                return int(token)
        raise DecodingError('No schema-valid continuation exists')

    def advance(self, token):
        result = self.transition(token)
        if result is None:
            raise DecodingError('Invalid chosen token')
        self.json, self.mode = result
        self.transitions.clear()
