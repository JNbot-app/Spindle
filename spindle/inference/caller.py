"""Tool calling with the native engine: prompt, grammar-constrained greedy decoding, result."""
import copy
import json
import math
import re
import threading
import time

import numpy as np

from spindle.prompt import render_prompt, render_tools
from spindle.tokenizer import Tokenizer
from .engine import Engine
from .filters import filter_calls
from .grammar import IncrementalSampler, expand_refs


def snake(name):
    """getWeather / Get-Weather -> get_weather. The model sees snake_case names only."""
    value = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
    value = re.sub(r'([A-Z])([A-Z][a-z])', r'\1_\2', value)
    return re.sub(r'[^\w]+', '_', value).strip('_').lower()


def normalize_schema(schema):
    """(schema with snake_case property names, {snake name: (original name, child aliases)})."""
    if isinstance(schema, bool):
        return schema, {}
    if not isinstance(schema, dict):
        raise ValueError('A JSON schema must be an object or a boolean')
    result, aliases = copy.deepcopy(schema), {}
    if result.get('properties') is not None:
        properties = {}
        for name, child in result['properties'].items():
            key = snake(name)
            if not key or key in properties:
                raise ValueError(f'Property names collide after normalization: {name}')
            properties[key], children = normalize_schema(child)
            aliases[key] = (name, children)
        result['properties'] = properties
        if 'required' in result:
            result['required'] = [snake(name) for name in result['required']]
    if ('properties' in result or result.get('type') == 'object') and 'additionalProperties' not in result:
        result['additionalProperties'] = False
    if isinstance(result.get('additionalProperties'), dict):
        result['additionalProperties'], _ = normalize_schema(result['additionalProperties'])
    if isinstance(result.get('items'), dict):
        result['items'], aliases['[]'] = normalize_schema(result['items'])
    for keyword in ('oneOf', 'anyOf', 'allOf'):
        if keyword in result:
            branches = []
            for branch in result[keyword]:
                normalized, children = normalize_schema(branch)
                branches.append(normalized)
                _merge_aliases(aliases, children)
            result[keyword] = branches
    if 'prefixItems' in result:
        items = []
        for index, child in enumerate(result['prefixItems']):
            normalized, aliases[f'[{index}]'] = normalize_schema(child)
            items.append(normalized)
        result['prefixItems'] = items
    return result, aliases


def _merge_aliases(target, source):
    for key, value in source.items():
        if key not in target:
            target[key] = value
        elif key.startswith('['):
            _merge_aliases(target[key], value)
        elif target[key][0] != value[0]:
            raise ValueError('Property aliases collide across schema variants')
        else:
            _merge_aliases(target[key][1], value[1])


def restore(value, aliases):
    """Map snake_case keys in a generated value back to the caller's original names."""
    if isinstance(value, list):
        return [restore(item, aliases.get(f'[{i}]', aliases.get('[]', {}))) for i, item in enumerate(value)]
    if isinstance(value, dict):
        return {aliases.get(key, (key, {}))[0]: restore(item, aliases.get(key, (key, {}))[1])
                for key, item in value.items()}
    return value


def constrained_log_probability(logits, token, sampler, top_k=32, window=12.0):
    """log p(token) renormalised over the grammar-valid tokens among the top k candidates."""
    values = np.asarray(logits, dtype=np.float64)
    full = float(values[token] - np.logaddexp.reduce(values))
    if full > -1e-4:
        return full
    close = np.flatnonzero(values >= values[token] - window)
    if len(close) > top_k:
        close = close[np.argpartition(-values[close], top_k - 1)[:top_k]]
    allowed = {int(t) for t in close if sampler.transition(int(t)) is not None} | {token}
    return float(values[token] - np.logaddexp.reduce(values[sorted(allowed)]))


class _CallTracker:
    """Maps generated JSON bytes to the index of the call they belong to."""

    def __init__(self):
        self.depth, self.in_string, self.escape, self.call = 0, False, False, -1

    def feed(self, data):
        touched = set()
        for byte in data:
            if self.in_string:
                if self.escape:
                    self.escape = False
                elif byte == 0x5c:
                    self.escape = True
                elif byte == 0x22:
                    self.in_string = False
            elif byte == 0x22:
                self.in_string = True
            elif byte in b'{[':
                self.depth += 1
                if byte == 0x7b and self.depth == 2:
                    self.call += 1
            elif byte in b'}]':
                if self.depth >= 2:
                    touched.add(self.call)
                self.depth -= 1
                continue
            if self.depth >= 2:
                touched.add(self.call)
        return touched


class Spindle:
    """A tool-calling model. It decides which tools to call and fills their arguments.

        with Spindle('model.cact', tools) as model:
            result = model('Turn off the kitchen lights')
            result['function_calls']  # [{'name': ..., 'arguments': {...}}]

    Tools are JSON-schema function definitions ({name, description, parameters}, or the OpenAI
    {"type": "function", "function": {...}} form). The model never runs a tool itself.

    Every call gets a probability: the mean grammar-constrained token probability of that call.
    With `filters` on (the default), calls the request does not support move to
    `suppressed_calls`, with the reason in `suppressed_reasons` (see filters.py). Calls below
    `min_call_probability` are suppressed too (0 keeps every call).

    `window` sets the attention window in tokens (0: the model's own, 512). A larger window helps
    when long tool lists push the needed tool out of view, at some extra latency. The tool part of
    the prompt is cached, so requests with the same tools skip most of its prefill.
    """

    def __init__(self, model, tools=None, *, system='', library=None, threads=4, filters=True,
                 min_call_probability=0.0, window=0):
        self.engine = Engine(model, library=library, threads=threads, window=window)
        self.tokenizer = Tokenizer.from_archive(model)
        self.filters = filters
        self.min_call_probability = min_call_probability
        self._lock = threading.RLock()
        self.set_tools(tools or [], system=system)

    def set_tools(self, tools, *, system=''):
        """Replace the tools and system prompt without reloading the model."""
        if isinstance(tools, str):
            tools = json.loads(tools)
        normalized, aliases = [], {}
        for entry in tools:
            entry = entry.get('function', entry) if entry.get('type') == 'function' else entry
            name = snake(entry['name'])
            if not name or name in aliases:
                raise ValueError(f'Tool names collide after normalization: {entry["name"]}')
            parameters, property_aliases = normalize_schema(
                expand_refs(entry.get('parameters', {'type': 'object', 'properties': {}})))
            normalized.append({'name': name, 'description': entry.get('description', ''), 'parameters': parameters})
            aliases[name] = (entry['name'], property_aliases)
        IncrementalSampler(normalized, self.tokenizer)  # validate the schemas now, not mid-request
        prefix = 1 + len(self.tokenizer.encode(render_tools(normalized, system)))
        with self._lock:
            self.tools, self.aliases, self.system = normalized, aliases, system.strip()
            self.engine.set_prefix(prefix)

    def __call__(self, query, max_new_tokens=256):
        if not isinstance(query, str):
            raise TypeError('The query must be text')
        with self._lock:
            started = time.perf_counter()
            prompt = render_prompt(self.tools, query, self.system)
            ids = [self.tokenizer.bos_id] + self.tokenizer.encode(prompt)
            if len(ids) + max_new_tokens > self.engine.context:
                raise ValueError('Prompt plus generation budget exceeds the context length')
            sampler = IncrementalSampler(self.tools, self.tokenizer)
            tracker, log_probs = _CallTracker(), {}
            self.engine.reset()
            logits = self.engine.eval(ids)
            prefill = time.perf_counter() - started
            generated = []
            for step in range(max_new_tokens):
                token = sampler.open_id if step == 0 else sampler.select_token(logits)
                if sampler.mode == 'json' and token != sampler.close_id:
                    value = constrained_log_probability(logits, token, sampler)
                    for index in tracker.feed(sampler.pieces[token]):
                        log_probs.setdefault(index, []).append(value)
                sampler.advance(token)
                generated.append(token)
                if sampler.mode == 'done':
                    break
                logits = self.engine.eval([token])
            complete = sampler.mode == 'done'
            calls = sampler.json.value if complete else []
            probabilities = [math.exp(sum(v) / len(v)) for v in
                             (log_probs.get(i, [0.0]) for i in range(len(calls)))]
            if self.filters:
                kept, suppressed, reasons = filter_calls(
                    calls, self.tools, query, system=self.system, call_probabilities=probabilities,
                    min_call_probability=self.min_call_probability)
            else:
                kept, suppressed, reasons = calls, [], []
            return {'function_calls': self._restore(kept), 'suppressed_calls': self._restore(suppressed),
                    'suppressed_reasons': reasons,
                    'call_probabilities': probabilities, 'confidence': self.engine.confidence(),
                    'complete': complete,
                    'metrics': {'prompt_tokens': len(ids), 'cached_prompt_tokens': self.engine.prefix_reused(),
                                'generated_tokens': len(generated),
                                'prefill_seconds': prefill, 'total_seconds': time.perf_counter() - started}}

    def _restore(self, calls):
        """Give calls back the caller's original tool and argument names."""
        result = []
        for call in calls:
            original, property_aliases = self.aliases[call['name']]
            result.append({'name': original, 'arguments': restore(call['arguments'], property_aliases)})
        return result

    def close(self):
        self.engine.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
