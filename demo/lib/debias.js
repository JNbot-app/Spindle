// Enum debiasing: correct the model's pull towards some labels of a closed list (port of spindle/inference/debias.py).
// At the first token of an enum value (each label starting with its own token), the model's probabilities over
// those first tokens are recorded per field and label list. After MIN_REQUESTS requests the label is picked by
// p(label) / prior(label), where the prior is the mean of those probabilities over earlier requests.
// It assumes labels come up about equally often over time.
import { variants } from './grammar.js';

const MIN_REQUESTS = 8;

export class EnumDebias {
  constructor(tokenizer) {
    this.tokenizer = tokenizer;
    this.priors = new Map(); // JSON [field, labels] -> { sum: Float64Array, count }
    this.firsts = new Map(); // JSON labels -> first token per label, or null when two labels share one
  }

  firstTokens(labels) {
    const key = JSON.stringify(labels);
    if (!this.firsts.has(key)) {
      const firsts = labels.map((label) => (label ? this.tokenizer.encode(label)[0] : undefined));
      const ok = !firsts.includes(undefined) && new Set(firsts).size === firsts.length;
      this.firsts.set(key, ok ? firsts : null);
    }
    return this.firsts.get(key);
  }

  /** [field, labels] when `state` sits right after the opening quote of an enum-only string value. */
  static enumStart(state) {
    if (state.mode !== 'string' || state.stringKey || state.raw.length || !state.stack.length) return null;
    const labels = [];
    for (const option of variants(state.scalarSchema)) {
      const values = option.enum ?? ('const' in option ? [option.const] : null);
      if (values === null) return null;
      for (const value of values) if (typeof value === 'string' && !labels.includes(value)) labels.push(value);
    }
    return labels.length > 1 ? [state.stack[state.stack.length - 1].key, labels] : null;
  }

  /** The debiased token at the start of an enum value, or null to let the sampler choose. */
  select(sampler, logits) {
    const start = sampler.mode === 'json' ? EnumDebias.enumStart(sampler.json) : null;
    const firsts = start && this.firstTokens(start[1]);
    if (!firsts || firsts.some((token) => sampler.transition(token) === null)) return null;
    const z = firsts.map((token) => logits[token]);
    const max = Math.max(...z);
    const p = z.map((v) => Math.exp(v - max));
    const sum = p.reduce((a, b) => a + b, 0);
    for (let i = 0; i < p.length; i++) p[i] /= sum;
    const key = JSON.stringify(start);
    if (!this.priors.has(key)) this.priors.set(key, { sum: new Float64Array(p.length), count: 0 });
    const total = this.priors.get(key);
    let choice = null;
    if (total.count >= MIN_REQUESTS) {
      let best = -Infinity;
      for (let i = 0; i < p.length; i++) {
        const score = Math.log(p[i]) - Math.log(Math.max(total.sum[i] / total.count, 1e-6));
        if (score > best) { best = score; choice = firsts[i]; }
      }
    }
    for (let i = 0; i < p.length; i++) total.sum[i] += p[i]; // this request joins the prior only after its own choice
    total.count++;
    return choice;
  }
}
