// Copy guard: keep a value that is being copied from the request word for word (port of spindle/inference/copying.py).
// When the string written so far appears in the request and the model picks a token that breaks the copy (and does
// not end the string), the token that continues the request is taken instead, if the model finds it almost as likely
// (within `margin` in log probability). Ending the string is never blocked and whitespace-only tokens are never forced.
import { topK } from './grammar.js';

const MIN_COPIED = 3; // bytes that must already match the request before the guard acts
const TOP = 32;       // candidate tokens looked at
const QUOTE = 0x22, BACKSLASH = 0x5c;
const SPACES = new Set([0x20, 0x09, 0x0a, 0x0b, 0x0c, 0x0d]);

function indexOf(haystack, needle) {
  outer: for (let i = 0; i + needle.length <= haystack.length; i++) {
    for (let j = 0; j < needle.length; j++) if (haystack[i + j] !== needle[j]) continue outer;
    return i;
  }
  return -1;
}

const continues = (raw, piece, query) => indexOf(query, [...raw, ...piece]) >= 0;

/** `chosen`, or a token that keeps copying the request. `query` is the request as UTF-8 bytes. */
export function copyGuard(sampler, logits, chosen, query, margin) {
  const state = sampler.json;
  if (sampler.mode !== 'json' || state.mode !== 'string' || state.stringKey || state.raw.length < MIN_COPIED
      || state.raw.includes(BACKSLASH) || indexOf(query, state.raw) < 0) return chosen;
  let piece = sampler.pieces[chosen];
  if (piece[0] === QUOTE || continues(state.raw, piece, query)) return chosen;
  const best = logits[chosen];
  for (const token of topK(logits, TOP)) {
    if (logits[token] < best - margin) break;
    piece = sampler.pieces[token];
    if (piece.some((b) => !SPACES.has(b)) && !piece.includes(QUOTE) && continues(state.raw, piece, query)
        && sampler.transition(token) !== null) return token;
  }
  return chosen;
}
