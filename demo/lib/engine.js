// The native engine compiled to WebAssembly (see native/wasm/build.sh), behind the interface
// that Spindle (caller.js) expects: eval, reset, setPrefix, prefixReused, confidence, setDecodeSkip.
import { readArchive, parseTokenizerBlob, Tokenizer } from './tokenizer.js';

const ERROR_BYTES = 2048;

/** Load a model from an ArrayBuffer. Returns { engine, tokenizer }. */
export async function loadModel(createSpindle, buffer) {
  const { vocab, context, tokenizerBlob } = readArchive(buffer);
  const tokenizer = new Tokenizer(parseTokenizerBlob(tokenizerBlob));
  const wasm = await createSpindle();
  wasm.FS.writeFile('/model.cact', new Uint8Array(buffer));
  const errorPtr = wasm._malloc(ERROR_BYTES);
  const fail = (what) => new Error(`${what}: ${wasm.UTF8ToString(errorPtr)}`);
  const path = wasm._malloc(16);
  wasm.HEAPU8.set(new TextEncoder().encode('/model.cact\0'), path);
  const handle = wasm._sp_load(path, errorPtr, ERROR_BYTES);
  wasm._free(path);
  if (!handle) throw fail('Could not load the model');
  wasm.FS.unlink('/model.cact'); // the engine keeps its own mapping

  const header = new Uint32Array(buffer, 0, 48);
  const layers = header[10], trainedMask = BigInt(header[45]) | BigInt(header[46]) << 32n;
  const skipLayers = [...Array(layers).keys()].filter((i) => (trainedMask >> BigInt(i)) & 1n);
  const tokenPtr = wasm._malloc(context * 4), logitPtr = wasm._malloc(vocab * 4), valuePtr = wasm._malloc(4);
  const engine = {
    context,
    vocab,
    skipLayers, // the layers this model was trained to skip at decode (empty for most models)
    eval(tokens) {
      if (tokens.length > context) throw new Error('Too many tokens');
      wasm.HEAPU8.length; // (heap views are refreshed by emscripten when memory grows)
      new Int32Array(wasm.HEAPU8.buffer, tokenPtr, tokens.length).set(tokens);
      if (wasm._np_eval(handle, tokenPtr, tokens.length, logitPtr, vocab, errorPtr, ERROR_BYTES) < 0) throw fail('Evaluation failed');
      return new Float32Array(wasm.HEAPU8.buffer, logitPtr, vocab).slice();
    },
    reset() { wasm._np_reset(handle); },
    setWindow(tokens) {
      if (wasm._np_set_window(handle, Math.min(tokens, context), errorPtr, ERROR_BYTES) < 0) throw fail('Could not set the window');
    },
    setPrefix(count) {
      if (wasm._np_set_prefix(handle, count, errorPtr, ERROR_BYTES) < 0) throw fail('Could not set the prefix');
    },
    // Decode skip: generated tokens skip these layers (off by default). Pass engine.skipLayers,
    // your own list, or [] to turn it off.
    setDecodeSkip(list) {
      const mask = (list || []).reduce((m, l) => m | (1n << BigInt(l)), 0n);
      if (wasm._np_set_decode_skip(handle, mask, errorPtr, ERROR_BYTES) < 0) throw fail('Could not set decode skip');
    },
    prefixReused() { return wasm._np_prefix_reused(handle); },
    confidence() {
      const present = wasm._np_confidence(handle, valuePtr, errorPtr, ERROR_BYTES);
      if (present < 0) throw fail('Confidence failed');
      return present ? new Float32Array(wasm.HEAPU8.buffer, valuePtr, 1)[0] : null;
    },
  };
  return { engine, tokenizer };
}
