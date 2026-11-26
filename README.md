# Spindle

[![Tests](https://github.com/JNbot-app/Spindle/actions/workflows/tests.yml/badge.svg)](https://github.com/JNbot-app/Spindle/actions/workflows/tests.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Demo](https://img.shields.io/badge/demo-in%20your%20browser-2a78d6.svg)](https://jnbot-app.github.io/Spindle/)

Spindle is a 75 MB tool-calling model (about 120M parameters, 4-bit weights) that runs locally on a CPU or in a browser tab. A 47 MB 2-bit version scores almost the same (see [Model files](#model-files)). You pass it your tool definitions as JSON Schema along with a user request, and it either picks the right tools and fills in their arguments, or returns nothing when no tool fits.

Because extraction and classification are just tool calls with the right schema, the same model handles those too. Grammar-constrained decoding ensures every call parses and matches your schema, and each prediction includes a probability so you can threshold uncertain results.

**[Try the live browser demo](https://jnbot-app.github.io/Spindle/)**. It downloads the model once and runs locally via WebAssembly with no server involved.

Spindle is built on [Needle 3](https://github.com/cactus-compute/needle) by Cactus Compute. See [the design notes](docs/changes-from-needle-3.md) for how the fine-tuning and runtime differ.

<p align="center"><img src="docs/images/pipeline.svg" alt="How a request flows through Spindle" width="840"></p>

## Quick start

You will need Python, CMake, and a C++17 compiler. On Linux, also install OpenBLAS (`apt install libopenblas-dev`).

Clone the repo, build the native engine, and download the weights:

```sh
git clone https://github.com/JNbot-app/Spindle
cd Spindle
pip install -e .
cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native -j
curl -L -o models/spindle.cact --create-dirs \
  https://github.com/JNbot-app/Spindle/releases/latest/download/spindle.cact
```

### Model files

| File | Size | Weights | Benchmark mean |
| --- | ---: | --- | ---: |
| `spindle.cact` | 75 MB | 4-bit | 51.1 |
| `spindle-2bit.cact` | 47 MB | 2-bit (E8 lattice), embedding and output head 4-bit | 50.8 |

Both files use the same API, prompt format and engine, so you can swap one path for the other. The 2-bit file is about 40% smaller to download and store, at a similar speed. It is a little weaker at extraction (DSTC8) and slightly stronger at declining (When2Call). To use it, download `spindle-2bit.cact` the same way:

```sh
curl -L -o models/spindle-2bit.cact --create-dirs \
  https://github.com/JNbot-app/Spindle/releases/latest/download/spindle-2bit.cact
```

### Python

```python
from spindle import Spindle

tools = [{
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "parameters": {
        "type": "object",
        "required": ["city"],
        "properties": {"city": {"type": "string"}},
    },
}]

with Spindle("models/spindle.cact", tools) as model:
    result = model("What's the weather in Lagos?")
    print(result["function_calls"])
    # [{'name': 'get_weather', 'arguments': {'city': 'Lagos'}}]
    print(result["call_probabilities"])

    # When no tool fits, function_calls is empty
    declined = model("Who won the 1998 World Cup?")
    print(declined["function_calls"])
    # []
```

### CLI

```sh
spindle run models/spindle.cact --tools examples/tools.json "Set a timer for 10 minutes"
```

Add `--metrics` to print token counts and timing.

## Extraction and classification

To extract structured fields from text, define a single tool whose `parameters` match the fields you want. To classify text, define a tool with an `enum` field. Pass `require_call=True` so the model does not decline:

```python
sentiment = [{
    "name": "classify_sentiment",
    "description": "Classify the sentiment of a text.",
    "parameters": {
        "type": "object",
        "required": ["sentiment"],
        "properties": {
            "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        },
    },
}]

with Spindle("models/spindle.cact", sentiment) as model:
    result = model("The battery died after two days.", require_call=True)
    print(result["function_calls"])
    # [{'name': 'classify_sentiment', 'arguments': {'sentiment': 'negative'}}]
```

## Runtime options

The tool prefix of the prompt is cached automatically, so repeated calls with the same `tools` list only evaluate the new user text.

- `system`: Pass a system prompt, such as `system="Current date and time: 2025-06-02T09:30"`, so the model can resolve relative dates like "tomorrow at noon".
- `window`: Context window size in tokens (default `1024`). Increase this if you have long tool lists, at the cost of higher latency.
- `require_call`: Set `True` to force a tool call instead of allowing an empty response.
- `filters`: Set `False` to bypass the output checks and return every decoded call, even ones the request may not fully support.
- `copy_margin`: Keeps copied argument values matching the input text word for word (default `4`, set `0` to disable).
- `debias_enums`: Corrects bias toward frequent labels in large closed `enum` lists using counts from earlier calls (default `True`). Set `False` if one label is naturally much more common in your workload.
- `decode_skip`: Set `True` (or `--decode-skip` in the CLI) to skip selected layers during token generation for faster decoding. This only works on models fine-tuned with `--skip-layers` (see [Fine-tuning](#fine-tuning)).

## Benchmarks

200 cases per set, scored identically across all models on an Apple M1 laptop. Needle 3 runs on its official engine, while LFM3, Qwen3.5, and FunctionGemma run on the GPU via `llama.cpp` with thinking turned off. See [docs/benchmarks.md](docs/benchmarks.md) for dataset details and scoring rules.

<p align="center"><img src="docs/images/benchmarks.svg" alt="Benchmark scores per model" width="760"></p>

| Benchmark | Metric | Spindle | Spindle 2-bit | Needle 3 | LFM3-1.7B | Qwen3.5-0.8B | FunctionGemma-270M |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mobile Actions | exact | 86.0 | 84.5 | 85.5 | **87.0** | 78.5 | 39.0 |
| DroidCall | exact | 35.5 | 37.0 | 41.0 | **50.0** | 19.0 | 25.0 |
| BFCL v4 | AST | 53.0 | 50.5 | 40.5 | **73.0** | 53.5 | 41.5 |
| When2Call | acc. | 67.5 | **72.5** | 64.5 | 68.0 | 62.5 | 64.5 |
| Tool removed | acc. | **73.0** | **73.0** | 38.0 | 65.5 | 26.0 | 43.0 |
| DSTC8 | F1 | 43.1 | 35.1 | 36.2 | **44.8** | 29.5 | 21.5 |
| SNIPS | F1 | 25.0 | 26.1 | 16.5 | **30.5** | 15.1 | 18.9 |
| SNIPS 7-way | F1 | 18.2 | 17.5 | 11.7 | **26.8** | 18.5 | 7.6 |
| Rotten Tomatoes | acc. | 65.5 | **67.5** | 48.5 | 15.5 | 27.0 | 10.0 |
| DBpedia | acc. | 44.0 | 44.5 | 31.0 | **79.5** | 53.0 | 2.0 |
| **Mean** | | 51.1 | 50.8 | 41.3 | **54.1** | 38.3 | 27.3 |
| Time per request (M1) | | 0.52 s | 0.46 s | 0.38 s | 9.36 s | 5.93 s | 0.57 s |
| File size | | 75 MB | 47 MB | 35 MB | 1.1 GB | 532 MB | 253 MB |

Spindle improves on Needle 3 across abstention (When2Call, Tool removed), larger tool lists (BFCL), extraction, and classification, while trailing on DroidCall where it declines more often and misses implicit details like MIME types. Compared to larger general models like LFM3-1.7B and Qwen3.5-0.8B, it gives up some accuracy on complex calls in exchange for a 75 MB footprint and ~0.5 s CPU latency.

<p align="center"><img src="docs/images/size.svg" alt="Mean score against model size" width="760"></p>

To run the benchmark suite yourself:

```sh
pip install -e '.[data]'
spindle bench models/spindle.cact --quick
spindle bench my-model --server http://127.0.0.1:8080/v1 --quick
```

## Running the browser demo locally

The [`demo/`](demo/) directory contains a static web app using the same C++ decoder compiled to WebAssembly. With `models/spindle.cact` downloaded, serve the repository root and open `http://localhost:8000/demo/`:

```sh
python3 -m http.server 8000
```

## Fine-tuning

Training data uses JSONL with one example per line (see [`examples/train.jsonl`](examples/train.jsonl)). Use the same JSON Schema format for `tools`, and leave `answers` empty when no tool fits:

```json
{"query": "Turn the lights off in the kitchen", "tools": [...], "answers": [{"name": "set_lights", "arguments": {"room": "kitchen", "on": false}}]}
{"query": "What is the capital of Peru?", "tools": [...], "answers": []}
```

Install the training extras and run fine-tuning (`'.[train]'` installs CPU JAX, use `'.[cuda]'` for NVIDIA GPUs):

```sh
pip install -e '.[train]'
spindle finetune --model models/spindle.cact --train train.jsonl --output runs/my-run
```

The output model is saved to `runs/my-run/model.cact`. Run `spindle finetune -h` for all options. Useful flags:

- `--updates-per-round 1000`: Saves a checkpoint and `progress.json` every 1,000 updates. Resume an interrupted run with `--resume runs/my-run`, or set `--max-minutes` to stop cleanly before a time limit.
- `--tool-variety` / `--absent-tools`: Varies tool lists across rows and creates negative examples by removing the needed tool, so the model checks the provided tools rather than guessing from the prompt. Set to `0` to disable.
- `--skip-layers`: Trains the model to skip specific layers during generation (for example `--skip-layers 5-8,10,13,15`). Find good candidate layers for your model with `spindle pick-skip model.cact`. On an earlier checkpoint, skipping layers gave roughly 1.4x faster generation for about 1.6 benchmark points.
- `spindle build-data --output data/general`: Rebuilds the general public training mix used for the released checkpoint.

## Model details and limitations

The released model is fine-tuned from Needle 3 for 1,000 updates (32k rows, batch size 32, roughly two hours on two T4 GPUs), with its call / no-call decisions distilled from a reasoning-trained teacher. The 2-bit model is trained the same way, but with its weights rounded to 2 bits during training. Most matrices use an E8 lattice code: each group of 8 rotated weights is stored as one 16-bit code (about 2 bits per weight) plus one scale per 128 weights, which cuts rounding error by about 10% compared with plain 2-bit levels. The embedding, the output head and a few late layers stay 4-bit. Fine-tuning `spindle-2bit.cact` keeps the 2-bit format automatically. The training mix combines public tool-calling, abstention, extraction, and classification datasets (none of which overlap with the benchmark sets). If you use the training mix or rebuild it with `spindle build-data`, check each source dataset's license before commercial use.

Known limitations:

- **Little world knowledge**: Spindle copies values from the prompt reliably, but will not infer unstated facts like airport codes or file MIME types.
- **Long tool lists**: Prompts exceeding the default 1024-token `window` push earlier tools out of context. Keep tool descriptions concise or raise `window`.
- **Multiple calls and optional fields**: Requests that require calling several different tools at once are the weakest case, and optional parameters can be skipped when many are defined together. Clear phrasing in the prompt helps.
- **Conservative defaults**: Output filters drop unsupported calls, which lowers false positives at the cost of some recall. Pass `filters=False` to inspect all raw decoded calls.
- **English only**.

## License

Apache 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Model weights are derived from [Needle 3](https://github.com/cactus-compute/needle), also licensed under Apache 2.0.