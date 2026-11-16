// Run every demo example through the model: node demo/node-examples.mjs model.cact [out.json]
import { readFileSync, writeFileSync } from 'node:fs';
import createSpindle from './wasm/spindle.js';
import { loadModel } from './lib/engine.js';
import { Spindle } from './lib/caller.js';
import { TASKS } from './examples.js';

const [modelPath, outPath] = process.argv.slice(2);
const buffer = readFileSync(modelPath);
const { engine, tokenizer } = await loadModel(createSpindle, buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength));
const model = new Spindle(engine, tokenizer);
const rows = [];
for (const task of TASKS) {
  for (const set of task.sets) {
    model.setTools(set.tools);
    for (const example of set.examples) {
      const r = model.run(example.text);
      const row = { task: task.id, set: set.name, label: example.label, text: example.text,
        calls: r.functionCalls, held: r.suppressedCalls, reasons: r.suppressedReasons, seconds: +r.metrics.totalSeconds.toFixed(2),
        cached: r.metrics.cachedPromptTokens, tokens: r.metrics.generatedTokens };
      rows.push(row);
      console.log(`[${set.name}] ${example.label} (${row.seconds}s, ${row.tokens} tok, cached ${row.cached})\n  -> ${JSON.stringify(r.functionCalls)}${r.suppressedCalls.length ? '\n  held: ' + JSON.stringify(r.suppressedCalls) + ' ' + r.suppressedReasons : ''}`);
    }
  }
}
if (outPath) writeFileSync(outPath, JSON.stringify(rows, null, 1));
