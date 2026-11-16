// Grammar-constrained decoding (port of spindle/inference/grammar.py).
// The output must be a JSON array of {"name", "arguments"} objects that match one tool's schema.
// JsonState follows the JSON byte by byte; the sampler picks the most likely token whose bytes
// keep that state valid.
import { isValid } from './schema.js';

export class DecodingError extends Error {}
export class InvalidPrefix extends Error {}

const has = (object, key) => Object.hasOwn(object, key);
const clone = (value) => structuredClone(value);
const decoder = new TextDecoder('utf-8', { fatal: true });
const encoder = new TextEncoder();

export function expandRefs(value, root = value, seen = []) {
  if (Array.isArray(value)) return value.map((v) => expandRefs(v, root, seen));
  if (value === null || typeof value !== 'object') return value;
  if ('$ref' in value) {
    const ref = value.$ref;
    if (typeof ref !== 'string' || !ref.startsWith('#/') || seen.includes(ref)) {
      throw new Error('Only non-recursive local JSON schema references are supported');
    }
    let target = root;
    for (const part of ref.slice(2).split('/')) {
      target = target?.[part.replaceAll('~1', '/').replaceAll('~0', '~')];
    }
    if (target === undefined) throw new Error('Unresolved local schema reference');
    const rest = Object.fromEntries(Object.entries(value).filter(([k]) => k !== '$ref'));
    return { allOf: [expandRefs(target, root, [...seen, ref]), expandRefs(rest, root, seen)] };
  }
  return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, expandRefs(v, root, seen)]));
}

/** Objects allow no extra keys unless the schema says otherwise. */
export function defaultAdditionalProperties(schema) {
  if (Array.isArray(schema)) return schema.map(defaultAdditionalProperties);
  if (schema === null || typeof schema !== 'object') return schema;
  const result = { ...schema };
  if (result.properties && typeof result.properties === 'object') {
    result.properties = Object.fromEntries(Object.entries(result.properties).map(([k, v]) => [k, defaultAdditionalProperties(v)]));
  }
  if ('items' in result) result.items = defaultAdditionalProperties(result.items);
  for (const key of ['prefixItems', 'oneOf', 'anyOf', 'allOf']) {
    if (Array.isArray(result[key])) result[key] = result[key].map(defaultAdditionalProperties);
  }
  for (const key of ['$defs', 'definitions']) {
    if (result[key] && typeof result[key] === 'object') {
      result[key] = Object.fromEntries(Object.entries(result[key]).map(([k, v]) => [k, defaultAdditionalProperties(v)]));
    }
  }
  const types = result.type;
  if (('properties' in result || types === 'object' || (Array.isArray(types) && types.includes('object'))) &&
      !('additionalProperties' in result)) result.additionalProperties = false;
  if (result.additionalProperties && typeof result.additionalProperties === 'object') {
    result.additionalProperties = defaultAdditionalProperties(result.additionalProperties);
  }
  return result;
}

export function variants(schema) {
  if (schema === false) return [];
  if (schema === true) return [{}];
  const branches = schema.oneOf ?? schema.anyOf ?? [schema];
  return branches.map((branch) => {
    const projected = { ...branch };
    for (let part of branch.allOf || []) {
      const options = variants(part);
      if (options.length !== 1) continue; // container completion still checks the full schema
      part = options[0];
      for (const key of ['type', 'items', 'prefixItems', 'enum', 'const']) {
        if (!(key in projected) && key in part) projected[key] = part[key];
      }
      if (part.properties) {
        const properties = { ...(projected.properties || {}) };
        for (const [key, value] of Object.entries(part.properties)) {
          properties[key] = has(properties, key) ? { allOf: [properties[key], value] } : value;
        }
        projected.properties = properties;
      }
      if ('additionalProperties' in part) projected.additionalProperties = part.additionalProperties;
      projected.required = [...new Set([...(projected.required || []), ...(part.required || [])])];
    }
    if (('properties' in projected || projected.type === 'object') && !('additionalProperties' in projected)) {
      projected.additionalProperties = false;
    }
    return projected;
  });
}

export function permits(schema, kind) {
  return variants(schema).some((item) => {
    const types = item.type;
    const allowed = typeof types === 'string' ? [types] : types;
    if (types === undefined || allowed.includes(kind) || (kind === 'number' && allowed.includes('integer'))) {
      return (item.allOf || []).every((part) => permits(part, kind));
    }
    return false;
  });
}

/** Legal JSON spellings of each character, including Unicode escapes. */
const ESCAPES = { '"': '\\"', '\\': '\\\\', '/': '\\/', '\b': '\\b', '\f': '\\f', '\n': '\\n', '\r': '\\r', '\t': '\\t' };
const encodingCache = new Map();
function stringEncodings(value) {
  if (encodingCache.has(value)) return encodingCache.get(value);
  const result = Array.from(value).map((character) => {
    const unicodeEscape = Array.from({ length: character.length }, (_, i) =>
      '\\u' + character.charCodeAt(i).toString(16).padStart(4, '0')).join('');
    const options = [[encoder.encode(unicodeEscape), true]];
    if (has(ESCAPES, character)) options.push([encoder.encode(ESCAPES[character]), false]);
    const code = character.codePointAt(0);
    if (code >= 32 && character !== '"' && character !== '\\' && !(code >= 0xd800 && code <= 0xdfff)) {
      options.push([encoder.encode(character), false]);
    }
    return options;
  });
  if (encodingCache.size > 512) encodingCache.clear();
  encodingCache.set(value, result);
  return result;
}

const startsWith = (bytes, prefix) => prefix.length <= bytes.length && prefix.every((b, i) => bytes[i] === b);
const lower = (bytes) => bytes.map((b) => (b >= 65 && b <= 70 ? b + 32 : b));

/** Keep prefixes only when an allowed string can still complete them. */
function stringPrefixMatches(raw, choices) {
  if (!raw.includes(0x5c)) {
    for (const choice of choices) if (startsWith(encoder.encode(choice), raw)) return true;
    return false;
  }
  for (const choice of choices) {
    let positions = new Set([0]);
    for (const encodings of stringEncodings(choice)) {
      const following = new Set();
      for (const position of positions) {
        const suffix = raw.slice(position);
        for (const [spelling, unicodeEscape] of encodings) {
          let fragment = suffix.slice(0, spelling.length);
          if (unicodeEscape) fragment = lower(fragment);
          if (startsWith(spelling, fragment)) {
            if (suffix.length <= spelling.length) return true;
            following.add(position + spelling.length);
          }
        }
      }
      positions = following;
      if (!positions.size) break;
    }
    if (positions.has(raw.length)) return true;
  }
  return false;
}

/** False when `bytes` can never become valid UTF-8 (a cut-off final character is fine). */
function utf8PrefixValid(bytes) {
  let i = 0;
  while (i < bytes.length) {
    const b = bytes[i];
    let need = 0, low = 0x80, high = 0xbf;
    if (b < 0x80) { i++; continue; }
    if (b >= 0xc2 && b <= 0xdf) need = 1;
    else if (b >= 0xe0 && b <= 0xef) { need = 2; if (b === 0xe0) low = 0xa0; if (b === 0xed) high = 0x9f; }
    else if (b >= 0xf0 && b <= 0xf4) { need = 3; if (b === 0xf0) low = 0x90; if (b === 0xf4) high = 0x8f; }
    else return false;
    for (let k = 1; k <= need; k++) {
      if (i + k >= bytes.length) return true;
      const c = bytes[i + k];
      if (c < (k === 1 ? low : 0x80) || c > (k === 1 ? high : 0xbf)) return false;
    }
    i += need + 1;
  }
  return true;
}

const INT = /^-?(?:0|[1-9][0-9]*)$/;
const INT_PREFIX = /^-?(?:0|[1-9][0-9]*)$/;
const NUM_PREFIX = /^-?(?:0|[1-9][0-9]*)(?:(?:\.[0-9]+)?(?:[eE][+-]?[0-9]*)?|\.[0-9]*)$/;
const NUM_FULL = /^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?$/;
const ascii = (bytes) => String.fromCharCode(...bytes);
const isDigit = (b) => b >= 48 && b <= 57;
const isHex = (b) => isDigit(b) || (b >= 97 && b <= 102) || (b >= 65 && b <= 70);

export class JsonState {
  constructor(schema) {
    this.schema = defaultAdditionalProperties(schema);
    this.stack = [];
    this.mode = null;
    this.raw = [];
    this.escape = false;
    this.unicodeRemaining = 0;
    this.scalarSchema = null;
    this.stringKey = false;
    this.done = false;
    this.value = null;
    this.lastWasSpace = false;
  }

  clone() {
    const other = Object.assign(Object.create(JsonState.prototype), this);
    other.raw = this.raw.slice();
    other.stack = this.stack.map((frame) => ({ ...frame, value: clone(frame.value) }));
    return other;
  }

  static isIntOnly(schema) {
    return permits(schema, 'integer') && !variants(schema).some((item) =>
      item.type === 'number' || (Array.isArray(item.type) && item.type.includes('number')));
  }

  static check(value, schema) {
    if (!isValid(value, defaultAdditionalProperties(schema))) throw new InvalidPrefix('Value violates schema');
  }

  effective(frame) {
    let options = variants(frame.schema);
    if (frame.kind === 'object' && has(frame.value, 'name')) {
      const selected = options.filter((s) => {
        const name = s.properties?.name;
        return !name || !('const' in name) || name.const === frame.value.name;
      });
      if (selected.length) options = selected;
    }
    return options;
  }

  availableKeys(frame) {
    const options = this.effective(frame);
    if (options.some((o) => (o.additionalProperties ?? false) !== false)) return null;
    const keys = new Set();
    for (const option of options) for (const key of Object.keys(option.properties || {})) if (!has(frame.value, key)) keys.add(key);
    return keys;
  }

  expected() {
    if (!this.stack.length) return this.schema;
    const frame = this.stack[this.stack.length - 1];
    const candidates = [];
    for (const item of this.effective(frame)) {
      if (frame.kind === 'array') {
        const position = frame.value.length, prefix = item.prefixItems || [];
        candidates.push(position < prefix.length ? prefix[position] : (item.items ?? {}));
      } else {
        candidates.push(has(item.properties || {}, frame.key) ? item.properties[frame.key] : (item.additionalProperties ?? false));
      }
    }
    const allowed = candidates.filter((s) => s !== false);
    if (!allowed.length) throw new InvalidPrefix('No allowed value');
    return allowed.length === 1 ? allowed[0] : { anyOf: allowed };
  }

  emit(value, schema) {
    JsonState.check(value, schema);
    if (!this.stack.length) { this.value = value; this.done = true; return; }
    const frame = this.stack[this.stack.length - 1];
    if (frame.kind === 'array') frame.value.push(value); else frame.value[frame.key] = value;
    frame.stage = 'delimiter';
  }

  close() {
    const frame = this.stack.pop();
    this.emit(frame.value, frame.schema);
  }

  byte(byte) {
    if (this.mode === 'string') return this.stringByte(byte);
    if (this.mode === 'number') {
      this.lastWasSpace = false;
      const intOnly = JsonState.isIntOnly(this.scalarSchema);
      const allowed = intOnly ? isDigit(byte) : (isDigit(byte) || '.eE+-'.includes(String.fromCharCode(byte)));
      if (allowed) {
        this.raw.push(byte);
        const text = ascii(this.raw);
        if (text !== '-' && !(intOnly ? INT_PREFIX : NUM_PREFIX).test(text)) throw new InvalidPrefix('Invalid number prefix');
        return;
      }
      const text = ascii(this.raw);
      if (!(intOnly ? INT : NUM_FULL).test(text)) throw new InvalidPrefix('Invalid number');
      this.mode = null;
      this.emit(Number(text), this.scalarSchema);
      return this.byte(byte);
    }
    if (this.mode === 'literal') {
      this.lastWasSpace = false;
      this.raw.push(byte);
      const text = ascii(this.raw);
      const targets = ['true', 'false', 'null'].filter((v) => v.startsWith(text));
      if (!targets.length) throw new InvalidPrefix('Invalid literal');
      if (targets.includes(text)) {
        this.mode = null;
        this.emit(JSON.parse(text), this.scalarSchema);
      }
      return;
    }
    if (byte === 32 || byte === 13 || byte === 10 || byte === 9) {
      if (byte !== 32 || this.lastWasSpace) throw new InvalidPrefix('Unnecessary whitespace');
      this.lastWasSpace = true;
      return;
    }
    this.lastWasSpace = false;
    if (this.done) throw new InvalidPrefix('Trailing data');
    if (this.stack.length) {
      const frame = this.stack[this.stack.length - 1], stage = frame.stage;
      if (stage === 'colon') {
        if (byte !== 58) throw new InvalidPrefix('Expected colon');
        frame.stage = 'value';
        return;
      }
      if (stage === 'delimiter') {
        if (byte === (frame.kind === 'object' ? 125 : 93)) { this.close(); return; }
        if (byte !== 44) throw new InvalidPrefix('Expected comma');
        if (frame.kind === 'object' && this.availableKeys(frame)?.size === 0) throw new InvalidPrefix('No further object properties are allowed');
        frame.stage = frame.kind === 'object' ? 'key' : 'value';
        return;
      }
      if (stage === 'key' || stage === 'key_or_end') {
        if (byte === 125 && stage === 'key_or_end') this.close();
        else if (byte === 34) {
          if (this.availableKeys(frame)?.size === 0) throw new InvalidPrefix('No object properties are allowed');
          this.mode = 'string'; this.raw = []; this.escape = false; this.stringKey = true;
        } else throw new InvalidPrefix('Expected object key');
        return;
      }
      if (stage === 'value_or_end' && byte === 93) { this.close(); return; }
    }
    const schema = this.expected();
    if (byte === 123 || byte === 91) {
      const kind = byte === 123 ? 'object' : 'array';
      if (!permits(schema, kind)) throw new InvalidPrefix('Wrong container type');
      this.stack.push({ schema, kind, value: kind === 'object' ? {} : [], stage: kind === 'object' ? 'key_or_end' : 'value_or_end', key: null });
    } else if (byte === 34) {
      if (!permits(schema, 'string')) throw new InvalidPrefix('Wrong string type');
      this.mode = 'string'; this.raw = []; this.escape = false; this.stringKey = false; this.scalarSchema = schema;
    } else if (byte === 45 || isDigit(byte)) {
      if (!permits(schema, 'number')) throw new InvalidPrefix('Wrong number type');
      this.mode = 'number'; this.raw = [byte]; this.scalarSchema = schema;
    } else if (byte === 116 || byte === 102 || byte === 110) {
      if (!permits(schema, byte === 110 ? 'null' : 'boolean')) throw new InvalidPrefix('Wrong literal type');
      this.mode = 'literal'; this.raw = [byte]; this.scalarSchema = schema;
    } else throw new InvalidPrefix('Expected JSON value');
  }

  stringByte(byte) {
    this.lastWasSpace = false;
    if (byte < 32) throw new InvalidPrefix('Control byte in string');
    if (byte === 34 && !this.escape) {
      let value;
      try { value = JSON.parse('"' + decoder.decode(Uint8Array.from(this.raw)) + '"'); }
      catch { throw new InvalidPrefix('Invalid JSON string'); }
      this.mode = null;
      if (this.stringKey) {
        const frame = this.stack[this.stack.length - 1];
        if (has(frame.value, value)) throw new InvalidPrefix('Duplicate key');
        frame.key = value; frame.stage = 'colon';
        this.expected();
      } else this.emit(value, this.scalarSchema);
      return;
    }
    if (this.unicodeRemaining) {
      if (!isHex(byte)) throw new InvalidPrefix('Invalid unicode escape');
      this.unicodeRemaining--;
    } else if (this.escape) {
      if (!'"\\/bfnrtu'.includes(String.fromCharCode(byte))) throw new InvalidPrefix('Invalid escape');
      this.unicodeRemaining = byte === 117 ? 4 : 0;
      this.escape = false;
    } else this.escape = byte === 92;
    this.raw.push(byte);
    if (this.stringKey) {
      const keys = this.availableKeys(this.stack[this.stack.length - 1]);
      if (keys !== null && !stringPrefixMatches(this.raw, keys)) throw new InvalidPrefix('Key cannot match an allowed property');
    }
    if (!this.raw.includes(92) && !utf8PrefixValid(this.raw)) throw new InvalidPrefix('Invalid UTF-8');
    if (!this.stringKey) {
      const choices = [];
      let unconstrained = false;
      for (const option of variants(this.scalarSchema)) {
        const values = option.enum ?? ('const' in option ? [option.const] : null);
        if (values === null) unconstrained = true;
        else choices.push(...values.filter((v) => typeof v === 'string'));
      }
      if (!unconstrained && !stringPrefixMatches(this.raw, choices)) throw new InvalidPrefix('String cannot match enum');
    }
  }

  feed(bytes) { for (const byte of bytes) this.byte(byte); }
}

export class IncrementalSampler {
  constructor(tools, tokenizer) {
    const alternatives = tools.map((tool) => ({
      type: 'object',
      properties: { name: { const: tool.name }, arguments: expandRefs(tool.parameters) },
      required: ['name', 'arguments'], additionalProperties: false,
    }));
    this.json = new JsonState({ type: 'array', items: alternatives.length ? { oneOf: alternatives } : false });
    this.mode = 'before'; // before -> json -> closed -> done
    this.tokenizer = tokenizer;
    this.openId = tokenizer.p2id.get('<tool_call>');
    this.closeId = tokenizer.p2id.get('</tool_call>');
    this.stopIds = new Set([tokenizer.eosId, tokenizer.p2id.get('<|im_end|>')]);
    this.pieces = tokenizer.pieceBytes();
    this.transitions = new Map();
  }

  /** [next JSON state, next mode] after `token`, or null if the token is not allowed. */
  transition(token) {
    if (this.transitions.has(token)) return this.transitions.get(token);
    let state = this.json, result;
    const mode = this.mode, special = this.tokenizer.isSpecial(token);
    if (this.stopIds.has(token)) result = mode === 'closed' ? [state, 'done'] : null;
    else if (mode === 'before') {
      result = token === this.openId ? [state, 'json'] : (token !== this.closeId && !special ? [state, mode] : null);
    } else if (mode === 'closed') result = null;
    else if (token === this.closeId) result = state.done ? [state, 'closed'] : null;
    else if (token === this.openId || special) result = null;
    else {
      state = state.clone();
      try { state.feed(this.pieces[token]); result = [state, mode]; }
      catch (error) {
        if (!(error instanceof InvalidPrefix)) throw error;
        result = null;
      }
    }
    this.transitions.set(token, result);
    return result;
  }

  /** The most likely allowed token: try the top 32 first, then the whole vocabulary. */
  selectToken(logits) {
    if (logits.length !== this.pieces.length) throw new DecodingError('Invalid decoder logits');
    const top = topK(logits, 32);
    for (const token of top) if (this.transition(token) !== null) return token;
    const all = Array.from({ length: logits.length }, (_, i) => i).sort((a, b) => logits[b] - logits[a] || a - b);
    for (const token of all) if (this.transition(token) !== null) return token;
    throw new DecodingError('No schema-valid continuation exists');
  }

  advance(token) {
    const result = this.transition(token);
    if (result === null) throw new DecodingError('Invalid chosen token');
    [this.json, this.mode] = result;
    this.transitions.clear();
  }
}

/** Indices of the k largest values, best first (ties: lower index first). */
export function topK(values, k) {
  const best = []; // sorted, best first
  let threshold = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (best.length === k && v <= threshold) continue;
    let at = best.length;
    while (at > 0 && values[best[at - 1]] < v) at--;
    best.splice(at, 0, i);
    if (best.length > k) best.pop();
    if (best.length === k) threshold = values[best[k - 1]];
  }
  return best;
}
