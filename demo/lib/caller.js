// Tool calling: prompt, grammar-constrained greedy decoding, result (port of spindle/inference/caller.py).
import { IncrementalSampler, expandRefs, topK } from './grammar.js';
import { copyGuard } from './copying.js';
import { filterCalls } from './filters.js';

const IM_START = '<|im_start|>', IM_END = '<|im_end|>';
const compact = (value) => (typeof value === 'string' ? value : JSON.stringify(value));

const STAMP = /(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})(?::\d{2})?/;
const WEEKDAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

/** The system prompt with its timestamp as the date fact the model was trained on:
 *  `date: YYYY-MM-DD Ddd HH:MM`, other lines kept after it. Unchanged without a timestamp
 *  or with its own `date:` fact. */
export function dateFact(system) {
  system = (system || '').trim();
  const match = STAMP.exec(system);
  if (!match || /(?<![A-Za-z])date:/.test(system)) return system;
  const [year, month, day, hour, minute] = match.slice(1).map(Number);
  const moment = new Date(Date.UTC(year, month - 1, day, hour, minute));
  if (moment.getUTCMonth() !== month - 1 || moment.getUTCDate() !== day || hour > 23 || minute > 59) return system;
  const rest = system.split('\n').map((line) => line.trim())
    .filter((line) => line && !STAMP.test(line) && !/^\W*day of (the )?week\b/i.test(line));
  const fact = `date: ${match[1]}-${match[2]}-${match[3]} ${WEEKDAYS[moment.getUTCDay()]} ${match[4]}:${match[5]}`;
  return [fact, ...rest].join('; ');
}

export function renderTools(tools, system = '') {
  system = dateFact(system);
  const head = system ? `${IM_START}system\n${system}${IM_END}\n` : '';
  return `${head}${IM_START}user\n<tools>${compact(tools)}</tools>\n`;
}
export const renderPrompt = (tools, query, system = '') =>
  `${renderTools(tools, system)}${query}${IM_END}\n${IM_START}assistant\n`;

/** getWeather / Get-Weather -> get_weather. The model sees snake_case names only. */
export function snake(name) {
  return name.replace(/([a-z0-9])([A-Z])/g, '$1_$2').replace(/([A-Z])([A-Z][a-z])/g, '$1_$2')
    .replace(/[^\p{L}\p{N}_]+/gu, '_').replace(/^_+|_+$/g, '').toLowerCase();
}

/** [schema with snake_case property names, {snake name: [original name, child aliases]}].
 *  Objects are closed (`additionalProperties: false`) unless the schema says otherwise: the grammar
 *  needs that, and the released model also calls more accurately when the prompt shows it. */
export function normalizeSchema(schema) {
  if (typeof schema === 'boolean') return [schema, {}];
  if (schema === null || typeof schema !== 'object') throw new Error('A JSON schema must be an object or a boolean');
  const result = structuredClone(schema), aliases = {};
  if (result.properties) {
    const properties = {};
    for (const [name, child] of Object.entries(result.properties)) {
      const key = snake(name);
      if (!key || key in properties) throw new Error(`Property names collide after normalization: ${name}`);
      let children;
      [properties[key], children] = normalizeSchema(child);
      aliases[key] = [name, children];
    }
    result.properties = properties;
    if (result.required) result.required = result.required.map(snake);
  }
  if (('properties' in result || result.type === 'object') && !('additionalProperties' in result)) result.additionalProperties = false;
  if (result.additionalProperties && typeof result.additionalProperties === 'object') {
    [result.additionalProperties] = normalizeSchema(result.additionalProperties);
  }
  if (result.items && typeof result.items === 'object') [result.items, aliases['[]']] = normalizeSchema(result.items);
  for (const keyword of ['oneOf', 'anyOf', 'allOf']) {
    if (!result[keyword]) continue;
    result[keyword] = result[keyword].map((branch) => {
      const [normalized, children] = normalizeSchema(branch);
      mergeAliases(aliases, children);
      return normalized;
    });
  }
  if (result.prefixItems) {
    result.prefixItems = result.prefixItems.map((child, i) => {
      const [normalized, children] = normalizeSchema(child);
      aliases[`[${i}]`] = children;
      return normalized;
    });
  }
  return [result, aliases];
}

function mergeAliases(target, source) {
  for (const [key, value] of Object.entries(source)) {
    if (!(key in target)) target[key] = value;
    else if (key.startsWith('[')) mergeAliases(target[key], value);
    else if (target[key][0] !== value[0]) throw new Error('Property aliases collide across schema variants');
    else mergeAliases(target[key][1], value[1]);
  }
}

/** Map snake_case keys in a generated value back to the caller's original names. */
function restore(value, aliases) {
  if (Array.isArray(value)) return value.map((item, i) => restore(item, aliases[`[${i}]`] ?? aliases['[]'] ?? {}));
  if (value !== null && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => {
      const [name, children] = aliases[key] ?? [key, {}];
      return [name, restore(item, children)];
    }));
  }
  return value;
}

const logSumExp = (values, indices) => {
  let max = -Infinity;
  for (const i of indices) if (values[i] > max) max = values[i];
  let sum = 0;
  for (const i of indices) sum += Math.exp(values[i] - max);
  return max + Math.log(sum);
};

/** log p(token) renormalised over the grammar-valid tokens among the top k candidates. */
function constrainedLogProbability(logits, token, sampler, topCount = 32, window = 12) {
  const all = { [Symbol.iterator]: function* () { for (let i = 0; i < logits.length; i++) yield i; } };
  const full = logits[token] - logSumExp(logits, all);
  if (full > -1e-4) return full;
  let close = [];
  for (let i = 0; i < logits.length; i++) if (logits[i] >= logits[token] - window) close.push(i);
  if (close.length > topCount) {
    close = topK(Float32Array.from(close, (i) => logits[i]), topCount).map((i) => close[i]);
  }
  const allowed = new Set(close.filter((t) => sampler.transition(t) !== null));
  allowed.add(token);
  return logits[token] - logSumExp(logits, allowed);
}

/** Maps generated JSON bytes to the index of the call they belong to. */
class CallTracker {
  constructor() { this.depth = 0; this.inString = false; this.escape = false; this.call = -1; }
  feed(bytes) {
    const touched = new Set();
    for (const byte of bytes) {
      if (this.inString) {
        if (this.escape) this.escape = false;
        else if (byte === 0x5c) this.escape = true;
        else if (byte === 0x22) this.inString = false;
      } else if (byte === 0x22) this.inString = true;
      else if (byte === 0x7b || byte === 0x5b) {
        this.depth++;
        if (byte === 0x7b && this.depth === 2) this.call++;
      } else if (byte === 0x7d || byte === 0x5d) {
        if (this.depth >= 2) touched.add(this.call);
        this.depth--;
        continue;
      }
      if (this.depth >= 2) touched.add(this.call);
    }
    return touched;
  }
}

/**
 * A tool-calling model. `engine` is a synchronous object with
 * eval(tokens) -> Float32Array logits, reset(), setPrefix(n), prefixReused(), confidence(), context.
 */
export class Spindle {
  constructor(engine, tokenizer, { filters = true, minCallProbability = 0, copyMargin = 4 } = {}) {
    this.engine = engine;
    this.tokenizer = tokenizer;
    this.filters = filters;
    this.minCallProbability = minCallProbability;
    this.copyMargin = copyMargin; // keep values copied from the request word for word (0: off), see copying.js
    this.setTools([]);
  }

  /** Replace the tools and system prompt without reloading the model. */
  setTools(tools, { system = '' } = {}) {
    const normalized = [], shown = [], aliases = {};
    for (let entry of tools) {
      if (entry.type === 'function' && entry.function) entry = entry.function;
      const name = snake(entry.name);
      if (!name || name in aliases) throw new Error(`Tool names collide after normalization: ${entry.name}`);
      const [parameters, propertyAliases] = normalizeSchema(expandRefs(entry.parameters ?? { type: 'object', properties: {} }));
      normalized.push({ name, description: entry.description ?? '', parameters });
      // The prompt shows each tool as written, with snake_case names and closed objects.
      const visible = { ...entry, name };
      if ('parameters' in entry) [visible.parameters] = normalizeSchema(entry.parameters);
      shown.push(visible);
      aliases[name] = [entry.name, propertyAliases];
    }
    new IncrementalSampler(normalized, this.tokenizer); // validate the schemas now, not mid-request
    const prefix = 1 + this.tokenizer.encode(renderTools(shown, system)).length;
    this.tools = normalized; this.promptTools = shown; this.aliases = aliases; this.system = system.trim();
    this.engine.setPrefix(prefix);
  }

  run(query, { maxNewTokens = 256, onToken = null } = {}) {
    const started = performance.now();
    const ids = [this.tokenizer.bosId, ...this.tokenizer.encode(renderPrompt(this.promptTools, query, this.system))];
    if (ids.length + maxNewTokens > this.engine.context) throw new Error('The input is too long for this model (try a shorter text)');
    const sampler = new IncrementalSampler(this.tools, this.tokenizer);
    const tracker = new CallTracker(), logProbs = new Map();
    this.engine.reset();
    let logits = this.engine.eval(ids);
    const prefill = (performance.now() - started) / 1000;
    const generated = [], queryBytes = new TextEncoder().encode(query);
    for (let step = 0; step < maxNewTokens; step++) {
      let token = step === 0 ? sampler.openId : sampler.selectToken(logits);
      if (step > 0 && this.copyMargin) token = copyGuard(sampler, logits, token, queryBytes, this.copyMargin);
      if (sampler.mode === 'json' && token !== sampler.closeId) {
        const value = constrainedLogProbability(logits, token, sampler);
        for (const index of tracker.feed(sampler.pieces[token])) {
          if (!logProbs.has(index)) logProbs.set(index, []);
          logProbs.get(index).push(value);
        }
      }
      sampler.advance(token);
      generated.push(token);
      if (onToken) onToken(generated.length);
      if (sampler.mode === 'done') break;
      logits = this.engine.eval([token]);
    }
    const complete = sampler.mode === 'done';
    const calls = complete ? sampler.json.value : [];
    const probabilities = calls.map((_, i) => {
      const values = logProbs.get(i) ?? [0];
      return Math.exp(values.reduce((a, b) => a + b, 0) / values.length);
    });
    const { kept, suppressed, reasons } = this.filters
      ? filterCalls(calls, this.tools, query, { system: this.system, callProbabilities: probabilities, minCallProbability: this.minCallProbability })
      : { kept: calls, suppressed: [], reasons: [] };
    const total = (performance.now() - started) / 1000;
    return {
      functionCalls: this.restoreCalls(kept), suppressedCalls: this.restoreCalls(suppressed), suppressedReasons: reasons,
      callProbabilities: probabilities, confidence: this.engine.confidence(), complete,
      metrics: { promptTokens: ids.length, cachedPromptTokens: this.engine.prefixReused(), generatedTokens: generated.length,
        prefillSeconds: prefill, totalSeconds: total },
    };
  }

  /** Read the tool part of the prompt ahead of time so the next request starts faster. */
  warm() {
    this.engine.reset();
    this.engine.eval([this.tokenizer.bosId, ...this.tokenizer.encode(renderPrompt(this.promptTools, 'hi', this.system))]);
  }

  restoreCalls(calls) {
    return calls.map((call) => {
      const [name, propertyAliases] = this.aliases[call.name];
      return { name, arguments: restore(call.arguments, propertyAliases) };
    });
  }
}
