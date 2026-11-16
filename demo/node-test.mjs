// Run the demo's pipeline in Node: node demo/node-test.mjs model.cact ["query" tools.json]
import { readFileSync } from 'node:fs';
import createSpindle from './wasm/spindle.js';
import { loadModel } from './lib/engine.js';
import { Spindle } from './lib/caller.js';

const [modelPath, query, toolsPath] = process.argv.slice(2);
const buffer = readFileSync(modelPath);
const started = performance.now();
const { engine, tokenizer } = await loadModel(createSpindle, buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength));
console.log(`loaded in ${((performance.now() - started) / 1000).toFixed(1)}s`);
const model = new Spindle(engine, tokenizer);
const tools = JSON.parse(readFileSync(toolsPath ?? new URL('../examples/tools.json', import.meta.url)));
model.setTools(tools);
const result = model.run(query ?? "What's the weather in Lagos?");
console.log(JSON.stringify(result, null, 1));
