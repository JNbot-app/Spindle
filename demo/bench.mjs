// Time the WebAssembly engine: node demo/bench.mjs model.cact [wasm/spindle.js] [rounds]
// Prefill reads a whole prompt with no cached prefix; decode feeds back the greedy token one at a time.
// The checksum of every logit lets you check that an engine change keeps the outputs identical.
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { loadModel } from './lib/engine.js';
import { Spindle, renderPrompt } from './lib/caller.js';

const [modelPath, enginePath = new URL('./wasm/spindle.js', import.meta.url).pathname, rounds = '3'] = process.argv.slice(2);
const { default: createSpindle } = await import(pathToFileURL(resolve(enginePath)).href);
const buffer = readFileSync(modelPath);
const { engine, tokenizer } = await loadModel(createSpindle, buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength));
const model = new Spindle(engine, tokenizer);
model.setTools(JSON.parse(readFileSync(new URL('../examples/tools.json', import.meta.url))));
engine.setPrefix(0); // measure the full prompt every time

const DECODE = 48;
const queries = ["What's the weather in Lagos?", 'Set a timer for 12 minutes called tea and tell me the weather in Oslo.'];
const median = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
let checksum = 0;
const fold = (logits) => { for (let i = 0; i < logits.length; i += 7) checksum = (checksum * 31 + Math.round(logits[i] * 1e4)) % 1e9; };
const argmax = (logits) => logits.reduce((best, v, i) => (v > logits[best] ? i : best), 0);

for (const query of queries) {
  const ids = [tokenizer.bosId, ...tokenizer.encode(renderPrompt(model.tools, query, ''))];
  const prefill = [], decode = [];
  for (let round = 0; round < Number(rounds); round++) {
    engine.reset();
    let started = performance.now();
    let logits = engine.eval(ids);
    prefill.push(performance.now() - started);
    if (round === 0) fold(logits);
    started = performance.now();
    for (let step = 0; step < DECODE; step++) {
      logits = engine.eval([argmax(logits)]);
      if (round === 0) fold(logits);
    }
    decode.push(performance.now() - started);
  }
  const p = median(prefill), d = median(decode);
  console.log(`${ids.length} prompt tokens: prefill ${p.toFixed(0)} ms (${(ids.length / p * 1000).toFixed(0)} tok/s), ` +
    `decode ${(d / DECODE).toFixed(1)} ms/token (${(DECODE / d * 1000).toFixed(1)} tok/s)`);
}
console.log(`checksum ${checksum}`);
