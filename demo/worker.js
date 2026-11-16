// Runs the model off the main thread so the page stays responsive while it thinks.
import createSpindle from './wasm/spindle.js';
import { loadModel } from './lib/engine.js';
import { Spindle } from './lib/caller.js';

let model = null, toolsKey = null, windowSize = 0;

// Apply the page's settings. A new window drops the cached prompt, so the tools are read again.
function useSettings({ window: size = 0, filters = true, minCallProbability = 0 } = {}) {
  if (size !== windowSize) { model.engine.setWindow(size); windowSize = size; toolsKey = null; }
  model.filters = filters;
  model.minCallProbability = minCallProbability;
}

function useTools(data) {
  useSettings(data.settings);
  const key = JSON.stringify([data.tools, data.system]);
  if (key !== toolsKey) { model.setTools(data.tools, { system: data.system }); toolsKey = key; }
}

self.onmessage = async ({ data }) => {
  const { type, id } = data;
  try {
    if (type === 'init') {
      const started = performance.now();
      const { engine, tokenizer } = await loadModel(createSpindle, data.model);
      model = new Spindle(engine, tokenizer);
      self.postMessage({ type: 'ready', seconds: (performance.now() - started) / 1000 });
    } else if (type === 'run') {
      useTools(data);
      const result = model.run(data.query, { maxNewTokens: data.settings?.maxNewTokens ?? 256, onToken: (tokens) => self.postMessage({ type: 'progress', id, tokens }) });
      self.postMessage({ type: 'result', id, result });
    } else if (type === 'warm') {
      useTools(data);
      model.warm();
    }
  } catch (error) {
    self.postMessage({ type: 'error', id, message: String(error?.message ?? error) });
  }
};
