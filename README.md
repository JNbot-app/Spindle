# Spindle

[![Tests](https://github.com/JNbot-app/Spindle/actions/workflows/tests.yml/badge.svg)](https://github.com/JNbot-app/Spindle/actions/workflows/tests.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Demo](https://img.shields.io/badge/demo-in%20your%20browser-2a78d6.svg)](https://jnbot-app.github.io/Spindle/)

**A 75 MB tool-calling model that runs on a phone CPU or in a browser tab.**

**[▶ Try the demo in your browser](https://jnbot-app.github.io/Spindle/)**: it downloads the model once and runs it
on your device. No server, no sign-up, and nothing you type leaves the page.

Give Spindle the tools your app exposes (as JSON Schema) and what the user said. It picks the right tools
and fills in their arguments, or says that none of them fits. The same model also extracts fields from text
and classifies it, because both are just a tool call with the right schema.

- **Small and local.** 4-bit weights, about 120M parameters. A request takes about half a second on a
  laptop CPU, with no GPU and no server. Nothing leaves the device.
- **Always valid output.** Decoding follows a grammar built from your schemas, so every call parses and fits
  its schema. Each call comes with a probability, so you can decide what to do with unsure ones.
- **Trained to say no.** When no listed tool fits, Spindle is trained to return no call instead of
  picking the nearest one. Its call / no-call choice is learned from a teacher that reasons before it decides.
- **Yours to fine-tune.** One command fine-tunes it on your own JSONL data; the released model was trained
  in about two hours on two T4 GPUs (not counting its teacher).

Spindle is built on [Needle 3](https://github.com/cactus-compute/needle) by Cactus Compute. It keeps the
architecture and changes how the model is fine-tuned and run: [what changed, and why](docs/changes-from-needle-3.md).

<p align="center"><img src="docs/images/pipeline.svg" alt="How a request flows through Spindle" width="840"></p>

## Results

<p align="center"><img src="docs/images/benchmarks.svg" alt="Benchmark scores per model" width="760"></p>

| Benchmark | Metric | Spindle | Needle 3 | LFM3-1.7B | Qwen3.5-0.8B | FunctionGemma-270M |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Mobile Actions | exact | 86.0 | 85.5 | **87.0** | 78.5 | 39.0 |
| DroidCall | exact | 33.5 | 41.0 | **50.0** | 19.0 | 25.0 |
| BFCL v4 | AST | 53.0 | 40.5 | **73.0** | 53.5 | 41.5 |
| When2Call | acc. | **70.0** | 64.5 | 68.0 | 62.5 | 64.5 |
| Tool removed | acc. | 53.5 | 38.0 | **65.5** | 26.0 | 43.0 |
| DSTC8 | F1 | 43.4 | 36.2 | **44.8** | 29.5 | 21.5 |
| SNIPS | F1 | 23.8 | 16.5 | **30.5** | 15.1 | 18.9 |
| SNIPS 7-way | F1 | 21.1 | 11.7 | **26.8** | 18.5 | 7.6 |
| Rotten Tomatoes | acc. | **65.0** | 48.5 | 15.5 | 27.0 | 10.0 |
| DBpedia | acc. | 46.5 | 31.0 | **79.5** | 53.0 | 2.0 |
| **Mean** | | 49.6 | 41.3 | **54.1** | 38.3 | 27.3 |
| Time per request (M1) | | 0.46 s | 0.38 s | 9.36 s | 5.93 s | 0.57 s |
| File size | | 75 MB | 35 MB | 1.1 GB | 532 MB | 253 MB |

200 cases per set, same cases and scoring for every model, on one Apple M1 laptop. Needle 3 runs on its own
official engine with its shipped file; the other three run in llama.cpp (on the GPU) with thinking off.
[How each set is scored](docs/benchmarks.md).

- **Against Needle 3**, the model it starts from, Spindle is better at declining when no tool fits
  (When2Call, Tool removed), at picking from larger tool lists (BFCL), at extraction and at classification
  (Rotten Tomatoes, DBpedia), and level on Mobile Actions. It is behind on DroidCall, mostly because it
  declines some requests that Needle 3 calls.
- **LFM3-1.7B**, a general model 15 times larger, scores higher on most sets, but takes about 20 times
  longer per request on the same laptop. Spindle is ahead on When2Call and Rotten Tomatoes (where LFM3 mostly
  wrote text instead of calling the tool).
- **Qwen3.5-0.8B** (7 times larger than Spindle) and **FunctionGemma-270M**, Google's small tool-calling model
  (3.4 times larger), score lower on average. Qwen is ahead of Spindle on BFCL and DBpedia, and takes
  about 13 times longer per request. FunctionGemma is about as fast, but copies the text into the enum
  field instead of choosing a label on classification (DBpedia 2.0).

<p align="center"><img src="docs/images/size.svg" alt="Mean score against model size" width="760"></p>

## Quick start

```sh
git clone https://github.com/JNbot-app/Spindle && cd Spindle
pip install -e .
cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release && cmake --build build/native -j
curl -L -o models/spindle.cact --create-dirs \
  https://github.com/JNbot-app/Spindle/releases/latest/download/spindle.cact
```

The engine needs CMake and a C++17 compiler; on Linux also OpenBLAS (`apt install libopenblas-dev`).

```python
from spindle import Spindle

tools = [{
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "required": ["city"],
                   "properties": {"city": {"type": "string"}}},
}]

with Spindle("models/spindle.cact", tools) as model:
    result = model("What's the weather in Lagos?")
    print(result["function_calls"])
    # [{'name': 'get_weather', 'arguments': {'city': 'Lagos'}}]
    print(result["call_probabilities"])
```

Or from the command line (`--metrics` adds token counts and timing):

```sh
spindle run models/spindle.cact --tools examples/tools.json "Set a timer for 10 minutes"
```

The tool part of the prompt is cached, so repeated requests with the same tools only process the new text.
Values copied from the request are kept word for word (the copy guard, `copy_margin=4`; `0` turns it off).
For closed label lists with many labels, enum debiasing (on by default) corrects the model's pull towards a few
labels, learned from earlier requests. It assumes the labels come up about equally often over time; if one label
is much more common in your use, turn it off with `debias_enums=False`.

A system prompt with the current date (`Spindle(..., system="Current date and time: 2025-06-02T09:30")`)
lets the model resolve "tomorrow at noon" and "next Tuesday".

**Faster decoding (decode skip, off by default).** Generated tokens can skip some layers; the prompt still
runs every layer. `spindle pick-skip model.cact` measures which layers a model can skip, and a short
`spindle finetune --skip-layers ...` run heals it; run the result with `decode_skip=True` (or `--decode-skip`).
On an earlier checkpoint this generated about 1.4 times faster for about 1.6 points less.
[How it works](docs/changes-from-needle-3.md#8-decode-skip).

## Extraction and classification

Both are tool calls. For extraction, give one tool whose parameters are the fields you want. For
classification, give one tool with an `enum` field. `require_call=True` makes sure the answer is not empty.

```python
sentiment = [{
    "name": "classify_sentiment",
    "description": "Classify the sentiment of a text.",
    "parameters": {"type": "object", "required": ["sentiment"], "properties": {
        "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]}}},
}]
with Spindle("models/spindle.cact", sentiment) as model:
    model("The battery died after two days.", require_call=True)["function_calls"]
    # [{'name': 'classify_sentiment', 'arguments': {'sentiment': 'negative'}}]
```

## Try it in the browser

**[Open the live demo](https://jnbot-app.github.io/Spindle/)**. The engine also compiles to WebAssembly, and
[`demo/`](demo/) is a static page with tool calling, extraction, sentiment and search-filter examples and a visual
tool editor. It runs entirely in the page, with the same decoding code as Python. It is published to GitHub Pages by
[`.github/workflows/demo.yml`](.github/workflows/demo.yml), with the model from the latest release (publish a
new release first, then push or run the workflow, to update the demo's model). To run it locally:

```sh
python3 -m http.server 8000     # from the repository root, with models/spindle.cact in place
# open http://localhost:8000/demo/
```

## Fine-tune on your own data

Training data is JSONL, one example per line (see [`examples/train.jsonl`](examples/train.jsonl)). An empty
`answers` list means "no tool fits this request".

```json
{"query": "Turn the lights off in the kitchen", "tools": [...], "answers": [{"name": "set_lights", "arguments": {"room": "kitchen", "on": false}}]}
```

```sh
pip install -e '.[train]'        # JAX on CPU; use '.[cuda]' on NVIDIA GPUs
spindle finetune --model models/spindle.cact --train train.jsonl --output runs/my-run
```

The result is `runs/my-run/model.cact`. `spindle finetune -h` lists every setting. A few worth knowing:

- **Rounds.** `--updates-per-round 1000` writes the model and a `progress.json` after every 1,000 updates.
  A run that stopped continues with `--resume runs/my-run`; `--max-minutes` stops before a round that would
  not fit in the time left.
- **Tool-list variety.** Some rows get a varied tool list, and copies of single-call rows without their
  needed tool are added with "no call" as the answer. This teaches the model to read the list instead of
  guessing from the request. Tune with `--tool-variety` and `--absent-tools` (0 turns each off).
- **Decode skip.** `--skip-layers 5-8,10,13,15` trains the model to run without those layers for the
  tokens it generates, and saves the list in the new file (run it with `--decode-skip`). Get a list for your
  model with `spindle pick-skip`. Fine-tuning a file that has a list keeps it; `--skip-layers none` drops it.
- **What is trained.** All weights, with their own learning rates for the backbone, the Monarch MLP factors
  and a runtime LoRA on attention. The 4-bit weights barely move under fine-tuning, so most of the learning
  goes into the FP16 parts; [the design notes](docs/changes-from-needle-3.md) explain why.

`spindle build-data --output data/general` rebuilds the general training mix from public datasets (below).

## Benchmarks

`spindle bench` scores a `.cact` model, or any OpenAI-compatible server, on ten public test sets with the
same cases and scoring. [docs/benchmarks.md](docs/benchmarks.md) describes each set, how every model was
run, and how to reproduce the numbers.

```sh
pip install -e '.[data]'
spindle bench models/spindle.cact --quick                         # 200 cases per set
spindle bench my-model --server http://127.0.0.1:8080/v1 --quick  # e.g. llama.cpp's llama-server
```

## The released model

| | |
| --- | --- |
| Parent | Needle 3, re-exported with 4-bit weights and converted to Monarch MLP factors (same outputs) |
| Size | 75 MB file; about 120M parameters, of which 6M are FP16 (Monarch factors and LoRA) |
| Context | 1024-token attention window by default (`window=`) |
| Training | 1,000 updates (32k rows, batch 32) from Needle 3, two NVIDIA T4 GPUs, about 2 hours; call / no-call choice distilled from a reasoning-trained teacher |
| Languages | English |

The training mix, built by `spindle build-data` from public data and templates (no text written by another
language model):

| Task | Share | Sources |
| --- | ---: | --- |
| Tool calling | 45% | [xLAM](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k), [ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE), [Glaive v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2), [CLINC150](https://huggingface.co/datasets/clinc/clinc_oos), [Banking77](https://huggingface.co/datasets/PolyAI/banking77), generated requests with relative dates, two requests joined into one (about 15% of the rows) |
| No fitting tool | 15% | [When2Call](https://huggingface.co/datasets/nvidia/When2Call), Glaive, CLINC150 out-of-scope, ToolACE, Banking77 |
| Extraction | 25% | [MASSIVE](https://huggingface.co/datasets/AmazonScience/massive), [Few-NERD](https://huggingface.co/datasets/DFKI-SLT/few-nerd), [Pile-NER](https://huggingface.co/datasets/Universal-NER/Pile-NER-type), [ATIS](https://huggingface.co/datasets/tuetschek/atis), generated records |
| Classification | 15% | [SST-2](https://huggingface.co/datasets/stanfordnlp/sst2), [TweetEval](https://huggingface.co/datasets/cardiffnlp/tweet_eval), [Emotion](https://huggingface.co/datasets/dair-ai/emotion), [AG News](https://huggingface.co/datasets/fancyzhx/ag_news), Banking77 |

None of the benchmark sets is in the training mix. Each dataset keeps its own license; check them before
commercial use.

### Limitations

- **Small model, little world knowledge.** Spindle copies values from the request well, but it cannot fill in
  facts the request does not state (a city's airport code, a file's MIME type).
- **Long tool lists.** Prompts beyond the 1024-token window push early tools out of view. Short tool
  descriptions help; so does a larger `window`, at some extra latency.
- **Several calls in one request.** Requests that need several different calls are the weakest case.
- **Optional details.** With many optional fields, the model sometimes leaves one out even when the text
  states it (a price limit next to a colour and a sort order, say). Clear wording ("below 80 euro") helps.
- **Caution over recall.** The output checks drop calls that the request does not support. That makes the
  model decline more, sometimes also when a call was wanted. `filters=False` returns every decoded call.
- **English only.**

## Repository layout

| Path | What it holds |
| --- | --- |
| [`spindle/inference/`](spindle/inference) | Engine binding, output checks, the `Spindle` class |
| [`spindle/model/`](spindle/model) | The network in JAX/Flax, 4-bit fake quantization, runtime LoRA |
| [`spindle/training/`](spindle/training) | Data preparation, span attention supervision, the fine-tuning loop |
| [`spindle/bench/`](spindle/bench) | Benchmark sets, scoring, and the runner behind `spindle bench` |
| [`spindle/datasets/`](spindle/datasets) | Converters for public datasets and the general training mix |
| [`spindle/archive.py`](spindle/archive.py) | Reading and writing `.cact` model files |
| [`native/`](native) | The C++ engine (NEON, AVX2, SSE2 and WebAssembly SIMD kernels) and the decoder (`grammar.cpp`: grammar, copy guard, enum debiasing), shared by Python and the demo |
| [`demo/`](demo) | The browser demo |
| [`benchmarks/`](benchmarks) | Benchmark results and the script that draws the charts |

## Acknowledgements

Spindle starts from the architecture and weights of [Needle 3](https://github.com/cactus-compute/needle) by
Cactus Compute, and is trained on the public datasets listed above. Thanks to their authors.

## License

Apache 2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). The model weights are derived from Needle 3, which
is also released under Apache 2.0.
