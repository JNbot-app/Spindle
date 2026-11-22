// The native engine compiled to WebAssembly (see native/wasm/build.sh), behind the interface
// that Spindle (caller.js) expects: eval, reset, setPrefix, prefixReused, confidence, setDecodeSkip, and
// createDecoder, the token choice (grammar, copy guard, enum debiasing) shared with Python (native/grammar.cpp).
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
    // Feed tokens after the cached ones; the logits for the next token stay in the engine (see createDecoder).
    feed(tokens) {
      if (tokens.length > context) throw new Error('Too many tokens');
      new Int32Array(wasm.HEAPU8.buffer, tokenPtr, tokens.length).set(tokens);
      if (wasm._np_eval(handle, tokenPtr, tokens.length, logitPtr, vocab, errorPtr, ERROR_BYTES) < 0) throw fail('Evaluation failed');
    },
    eval(tokens) {
      this.feed(tokens);
      return new Float32Array(wasm.HEAPU8.buffer, logitPtr, vocab).slice();
    },
    /** A decoder that picks each next token from the logits of the last `feed`. */
    createDecoder({ pieces, special, openId, closeId, thinkId = -1, stopIds }) {
      const sizes = pieces.map((p) => p.length), total = sizes.reduce((a, b) => a + b, 0);
      const bytesPtr = wasm._malloc(total), offsetsPtr = wasm._malloc((pieces.length + 1) * 4);
      const specialPtr = wasm._malloc(pieces.length), stopPtr = wasm._malloc(stopIds.length * 4);
      const offsets = new Uint32Array(wasm.HEAPU8.buffer, offsetsPtr, pieces.length + 1);
      let at = 0;
      pieces.forEach((piece, i) => { offsets[i] = at; wasm.HEAPU8.set(piece, bytesPtr + at); at += piece.length; });
      offsets[pieces.length] = at;
      wasm.HEAPU8.set(special, specialPtr);
      new Int32Array(wasm.HEAPU8.buffer, stopPtr, stopIds.length).set(stopIds);
      const decoder = wasm._np_decoder_new(bytesPtr, offsetsPtr, specialPtr, pieces.length, openId, closeId, thinkId,
        stopPtr, stopIds.length, errorPtr, ERROR_BYTES);
      for (const ptr of [bytesPtr, offsetsPtr, specialPtr, stopPtr]) wasm._free(ptr);
      if (!decoder) throw fail('Could not create the decoder');
      const withString = (text, use) => {
        const size = wasm.lengthBytesUTF8(text) + 1, ptr = wasm._malloc(size);
        wasm.stringToUTF8(text, ptr, size);
        try { return use(ptr, size - 1); } finally { wasm._free(ptr); }
      };
      return {
        setTools(tools, labelTokens) {
          const status = withString(JSON.stringify(tools), (toolsPtr) => withString(JSON.stringify(labelTokens),
            (labelsPtr) => wasm._np_decoder_set_tools(decoder, toolsPtr, labelsPtr, errorPtr, ERROR_BYTES)));
          if (status < 0) throw fail('Invalid tools');
        },
        start(query, { requireCall = false, reasoning = true, debias = true, copyMargin = 4 } = {}) {
          const flags = (requireCall ? 1 : 0) | (reasoning ? 2 : 0) | (debias ? 4 : 0);
          const status = withString(query, (ptr, size) => wasm._np_decoder_start(decoder, ptr, size, flags, copyMargin, errorPtr, ERROR_BYTES));
          if (status < 0) throw fail('Could not start decoding');
        },
        step() {
          const token = wasm._np_decoder_step(decoder, logitPtr, vocab, errorPtr, ERROR_BYTES);
          if (token < 0) throw fail('Decoding failed');
          return token;
        },
        get done() { return wasm._np_decoder_state(decoder) === 3; },
        result() {
          const size = Number(wasm._np_decoder_result(decoder, 0, 0)) + 1, ptr = wasm._malloc(size);
          try {
            wasm._np_decoder_result(decoder, ptr, size);
            return JSON.parse(wasm.UTF8ToString(ptr));
          } finally { wasm._free(ptr); }
        },
      };
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
