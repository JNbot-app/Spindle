# Browser demo

A static page that downloads the model and runs it in WebAssembly, with no server involved.

| Path | What it is |
| --- | --- |
| `index.html`, `style.css`, `app.js` | The page |
| `tools-editor.js`, `dom.js` | The visual tool editor and small DOM helpers |
| `examples.js` | The tasks, tool definitions and example inputs shown in the page |
| `worker.js` | Runs the model in a Web Worker so the page stays responsive |
| `lib/` | JavaScript port of the Python inference code: tokenizer, prompt, grammar-constrained decoding, output filters |
| `bench.mjs`, `node-test.mjs`, `node-examples.mjs` | Node scripts: engine speed, one request, every example |
| `wasm/` | The native engine compiled for the browser (built by `native/wasm/build.sh`) |
| `spindle.cact` | The model the page downloads (not in git: the deploy workflow takes it from the latest release) |

## Run it locally

```sh
cp models/spindle.cact demo/            # or skip this: the page falls back to ../models/spindle.cact
python3 -m http.server 8000                # from the repository root
# open http://localhost:8000/demo/
```

## Rebuild the engine

Needs [Emscripten](https://emscripten.org) (`brew install emscripten`).

```sh
native/wasm/build.sh
```

The browser build runs on one thread (GitHub Pages cannot send the headers that WebAssembly threads need).
It has its own WASM SIMD kernels (`native/kernels_wasm.cpp`, which look up 4-bit codes with byte shuffles) and
replaces BLAS with the register-blocked SIMD loops in `native/wasm/cblas.cpp`. Both keep the summation order of
plain loops, so a faster kernel gives the exact same outputs.

To time the engine (prefill and decode speed, plus a checksum of the logits to confirm a change keeps the outputs identical):

```sh
node demo/bench.mjs models/spindle.cact                      # the engine in demo/wasm
node demo/bench.mjs models/spindle.cact /path/to/spindle.js  # another build, to compare
```

## Check the JavaScript port

```sh
node demo/node-test.mjs models/spindle.cact "What's the weather in Lagos?"   # one request
node demo/node-examples.mjs models/spindle.cact                                # every example in the page
```

Results can differ from the Python package in rare near-tie cases, because the matrix products are summed in a
different order.

## Deploy

`.github/workflows/demo.yml` publishes `demo/` to GitHub Pages on every push to `main` that touches it and on
every new release. It downloads `spindle.cact` from the latest release, so the model never enters the git history.
In the repository settings, set **Pages > Source** to **GitHub Actions**. When you release a new model, also
change `CACHE_NAME` in `app.js`, so browsers that cached the old file download the new one.
