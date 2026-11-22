"""Tool calling with the native engine: prompt, grammar-constrained greedy decoding, result."""
import copy
import json
import re
import threading
import time

from jsonschema import Draft202012Validator

from spindle.prompt import THINK_END, THINK_START, render_prompt, render_tools
from spindle.tokenizer import Tokenizer
from .engine import Decoder, Engine
from .filters import filter_calls


def snake(name):
    """getWeather / Get-Weather -> get_weather. The model sees snake_case names only."""
    value = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
    value = re.sub(r'([A-Z])([A-Z][a-z])', r'\1_\2', value)
    return re.sub(r'[^\w]+', '_', value).strip('_').lower()


def normalize_schema(schema):
    """(schema with snake_case property names, {snake name: (original name, child aliases)}).

    Objects with listed properties are closed (`additionalProperties: false`) unless the schema says
    otherwise: the released model calls more accurately when the prompt shows it. A free-form object
    (`"type": "object"` without properties, a dict of any keys) stays open: closing it would only allow `{}`."""
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
    if 'properties' in result and 'additionalProperties' not in result:
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


def enum_labels(schema, found=None):
    """Every string in an `enum` or `const` of a schema (the labels enum debiasing may correct)."""
    found = set() if found is None else found
    if isinstance(schema, list):
        for item in schema:
            enum_labels(item, found)
    elif isinstance(schema, dict):
        for key, value in schema.items():
            if key == 'enum' and isinstance(value, list):
                found.update(v for v in value if isinstance(v, str))
            elif key == 'const' and isinstance(value, str):
                found.add(value)
            else:
                enum_labels(value, found)
    return found


def token_bytes(tokenizer):
    """The bytes each token stands for."""
    return [bytes([int(p[3:5], 16)]) if kind == 4 else p.replace('\u2581', ' ').encode('utf-8')
            for p, kind in zip(tokenizer.pieces, tokenizer.types)]


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

    `window` sets the attention window in tokens (0: the model's own, 1024 for the base model).
    `decode_skip` lets generated tokens skip some layers for faster decoding (off by default; see
    Engine.set_decode_skip): True uses the layers the model was trained to skip.
    `debias_enums` corrects the model's pull towards some labels of long enum lists, learned from
    earlier requests (on by default, False: off; it assumes labels come up about equally often).
    `copy_margin` (default 4, 0: off) keeps a value that is being copied from the request word for word when the
    model would reword it halfway and the copy is at most this much less likely.
    Token choice (grammar, copy guard, debiasing) runs in the native library (native/grammar.cpp).
    `threads` is the size of the worker team (0: half the performance cores, at most four). The
    tool part of the prompt is cached, so requests with the same tools skip most of its prefill.

    The prompt shows each tool as you wrote it (snake_case names, closed objects), and a timestamp in the
    system prompt is rewritten to the date fact the model was trained on (see prompt.py).

    With `reasoning` on (the default), a model that wants to write a short note before its calls
    may do so; the note is returned as `reasoning`. It makes copied values more exact and costs
    some extra tokens. Models trained without notes answer with calls right away.
    """

    def __init__(self, model, tools=None, *, system='', library=None, threads=0, filters=True,
                 min_call_probability=0.0, window=0, reasoning=True, decode_skip=None, debias_enums=True,
                 copy_margin=4.0):
        self.engine = Engine(model, library=library, threads=threads, window=window, decode_skip=decode_skip)
        self.tokenizer = Tokenizer.from_archive(model)
        self.filters = filters
        self.reasoning = reasoning
        self.debias_enums = debias_enums
        self.copy_margin = copy_margin
        tokenizer = self.tokenizer
        self.decoder = Decoder(self.engine.library, token_bytes(tokenizer), [t in (1, 2) for t in tokenizer.types],
                               tokenizer.p2id['<tool_call>'], tokenizer.p2id['</tool_call>'], tokenizer.p2id.get(THINK_START),
                               {tokenizer.eos_id, tokenizer.p2id['<|im_end|>']})
        self.min_call_probability = min_call_probability
        self._lock = threading.RLock()
        self._tools_key = None
        self.set_tools(tools or [], system=system)

    def set_tools(self, tools, *, system=''):
        """Replace the tools and system prompt without reloading the model."""
        if isinstance(tools, str):
            tools = json.loads(tools)
        key = json.dumps([tools, system], sort_keys=True, default=str)
        if key == self._tools_key:  # same tools as before: nothing to prepare again
            return
        normalized, shown, aliases = [], [], {}
        for entry in tools:
            entry = entry.get('function', entry) if entry.get('type') == 'function' else entry
            name = snake(entry['name'])
            if not name or name in aliases:
                raise ValueError(f'Tool names collide after normalization: {entry["name"]}')
            parameters, property_aliases = normalize_schema(
                expand_refs(entry.get('parameters', {'type': 'object', 'properties': {}})))
            normalized.append({'name': name, 'description': entry.get('description', ''), 'parameters': parameters})
            aliases[name] = (entry['name'], property_aliases)
            # The prompt shows the tool as written, with snake_case names and closed objects.
            visible = dict(entry, name=name)
            if 'parameters' in entry:
                visible['parameters'] = normalize_schema(entry['parameters'])[0]
            shown.append(visible)
        for tool in normalized:  # check the schemas now, not mid-request
            Draft202012Validator.check_schema(tool['parameters'])
        labels = {label: self.tokenizer.encode(label)[0] for label in enum_labels(normalized) if label}
        prefix = 1 + len(self.tokenizer.encode(render_tools(shown, system)))
        with self._lock:
            self.decoder.set_tools([{'name': t['name'], 'parameters': t['parameters']} for t in normalized], labels)
            self.tools, self.prompt_tools, self.aliases, self.system = normalized, shown, aliases, system.strip()
            self._tools_key = key
            self.engine.set_prefix(prefix)

    def __call__(self, query, max_new_tokens=256, require_call=False):
        """Answer one request. `require_call` forbids the empty answer: use it for extraction and
        classification, where some call is always right."""
        if not isinstance(query, str):
            raise TypeError('The query must be text')
        with self._lock:
            started = time.perf_counter()
            prompt = render_prompt(self.prompt_tools, query, self.system)
            ids = [self.tokenizer.bos_id] + self.tokenizer.encode(prompt)
            if len(ids) + max_new_tokens > self.engine.context:
                raise ValueError('Prompt plus generation budget exceeds the context length')
            self.decoder.start(query, require_call=require_call and bool(self.tools), reasoning=self.reasoning,
                               debias=self.debias_enums, copy_margin=self.copy_margin)
            self.engine.reset()
            logits = self.engine.eval(ids)
            prefill = time.perf_counter() - started
            generated = []
            for _ in range(max_new_tokens):
                generated.append(self.decoder.step(logits))
                if self.decoder.done:
                    break
                logits = self.engine.eval(generated[-1:])
            answer = self.decoder.result()
            open_id = self.tokenizer.p2id['<tool_call>']
            note = generated[1:generated.index(open_id)] if open_id in generated[1:] else []
            note = self.tokenizer.decode(note).replace(THINK_END, '').strip()
            calls, probabilities, complete = answer['calls'], answer['probabilities'], answer['complete']
            if self.filters:
                kept, suppressed, reasons = filter_calls(
                    calls, self.tools, query, system=self.system, call_probabilities=probabilities,
                    min_call_probability=self.min_call_probability)
            else:
                kept, suppressed, reasons = calls, [], []
            return {'function_calls': self._restore(kept), 'suppressed_calls': self._restore(suppressed),
                    'suppressed_reasons': reasons, 'reasoning': note,
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
        self.decoder.close()
        self.engine.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
