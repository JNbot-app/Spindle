// Output filters: hold back calls the request does not support (port of spindle/inference/filters.py).
//   grounding:   an argument value must come from the request text
//   negation:    "don't open the door" holds back a call that would open it
//   probability: calls below `minCallProbability` are held back
//   duplicates:  the same call twice is kept once
import { isValid } from './schema.js';

const NUMBER = /(?<![\p{L}\p{N}_.,])[-+]?\d+(?:[.,]\d+)*/u;
const NUMBER_WORD = /\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|half|quarter|a couple|a few)\b/i;
const DATE_FACT = /\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?/g;

// Enum values that are usually said with another word ("turn the light up" -> "on").
const SYNONYMS = {
  open: ['open', 'unlock', 'unlatch'], close: ['close', 'shut', 'lock'],
  on: ['on', 'enable', 'activate', 'start', 'brighten', 'dim'],
  off: ['off', 'disable', 'deactivate', 'stop'],
  forward: ['forward', 'forwards', 'ahead'], backward: ['backward', 'backwards', 'back'],
  left: ['left', 'counterclockwise'], right: ['right', 'clockwise'],
};
// Tools whose output is a judgement about the text (not copied from it) skip grounding.
const JUDGEMENT = /classif|sentiment|detect|categor|analy[sz]/i;
const ACTION_VERBS = new Set(['move', 'rotate', 'open', 'close', 'lock', 'unlock', 'send', 'delete', 'create',
  'start', 'stop', 'play', 'pause', 'buy', 'sell', 'turn', 'remove', 'cancel']);
const NEGATION = String.raw`\b(?:do\s+not|don['’]t|never|must\s+not)\s+(?:\w+\s+){0,2}`;

const words = (text) => (String(text).replaceAll('_', ' ').toLowerCase().match(/[\p{L}\p{N}_]+/gu) || []).join(' ');
export const mentioned = (value, source) => {
  const needle = words(value);
  return needle !== '' && ` ${words(source)} `.includes(` ${needle} `);
};
const isTemporal = (key, schema) => ['date', 'date-time', 'time'].includes(schema.format) || /date|time/.test(key);

class Grounding {
  constructor(source) {
    this.source = source;
    this.hasQuantity = NUMBER.test(source.replace(DATE_FACT, ' ')) || NUMBER_WORD.test(source);
    this.missing = [];
  }

  supported(key, value, schema) {
    if (typeof value === 'boolean' || (typeof value !== 'number' && typeof value !== 'string')) return true;
    if (typeof value !== 'string') return this.hasQuantity;
    if (schema.enum) return (SYNONYMS[value] || [value]).some((word) => mentioned(word, this.source));
    return isTemporal(key, schema) || mentioned(value, this.source);
  }

  check(value, schema, path = '') {
    if (schema === null || typeof schema !== 'object') return;
    const choices = schema.anyOf ?? schema.oneOf;
    if (choices) schema = { ...schema, ...(choices.find((s) => isValid(value, s)) || {}) };
    for (const part of schema.allOf || []) this.check(value, part, path);
    if (Array.isArray(value)) {
      const prefix = schema.prefixItems || [];
      value.forEach((item, i) => this.check(item, i < prefix.length ? prefix[i] : (schema.items ?? {}), `${path}[${i}]`));
    } else if (value !== null && typeof value === 'object') {
      const properties = schema.properties || {};
      for (const key of Object.keys(value)) {
        const child = properties[key];
        if (child === null || typeof child !== 'object') continue;
        const item = value[key], childPath = path ? `${path}.${key}` : key;
        if (item !== null && typeof item === 'object') this.check(item, child, childPath);
        else if (!this.supported(key, item, child)) {
          if (!(schema.required || []).includes(key)) delete value[key];
          else if ('default' in child) value[key] = structuredClone(child.default);
          else this.missing.push(childPath);
        }
      }
    }
  }
}

function negated(call, query) {
  const [verb, ...rest] = call.name.split('_');
  const verbs = new Set(ACTION_VERBS.has(verb) || (rest.length && verb !== 'get' && verb !== 'set') ? [verb] : []);
  for (const value of Object.values(call.arguments)) {
    if (typeof value === 'string' && Object.hasOwn(SYNONYMS, value)) SYNONYMS[value].forEach((v) => verbs.add(v));
  }
  return [...verbs].some((v) => new RegExp(NEGATION + v.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + String.raw`\b`, 'i').test(query));
}

/** Split decoded calls into { kept, suppressed, reasons } (snake_case names). */
export function filterCalls(calls, tools, query, { system = '', callProbabilities = null, minCallProbability = 0 } = {}) {
  const schemas = Object.fromEntries(tools.map((t) => [t.name, t.parameters ?? {}]));
  const source = query + '\n' + (system || '');
  const kept = [], suppressed = [], reasons = [], seen = new Set();
  structuredClone(calls).forEach((call, index) => {
    let reason = null;
    const probability = callProbabilities ? callProbabilities[index] : 1;
    if (probability < minCallProbability) reason = `probability ${probability.toFixed(2)} is below ${minCallProbability.toFixed(2)}`;
    else if (negated(call, query)) reason = 'the request says not to do this';
    else if (!JUDGEMENT.test(call.name)) {
      const grounding = new Grounding(source);
      grounding.check(call.arguments, schemas[call.name]);
      if (grounding.missing.length) reason = 'not in the request: ' + grounding.missing.join(', ');
    }
    if (reason) { suppressed.push(call); reasons.push(reason); return; }
    const key = JSON.stringify(call, (k, v) => (v && typeof v === 'object' && !Array.isArray(v)
      ? Object.fromEntries(Object.entries(v).sort(([a], [b]) => (a < b ? -1 : 1))) : v));
    if (!seen.has(key)) { seen.add(key); kept.push(call); }
  });
  return { kept, suppressed, reasons };
}
