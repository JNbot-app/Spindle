# Spindle

A small tool-calling model that runs on-device. Give it the tools your app exposes and a
request; it picks the right tools and fills in their arguments. Built on Needle 3.

## Install

```sh
pip install -e .             # inference
pip install -e '.[train]'    # + fine-tuning (JAX); use '.[cuda]' on NVIDIA GPUs
```

Build the native engine (needs CMake and a C++17 compiler; Linux also needs OpenBLAS):

```sh
cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native -j
```

## Run

```python
from spindle import Spindle

tools = [{"name": "get_weather", "description": "Get the current weather for a city.",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                         "required": ["city"]}}]

with Spindle("model.cact", tools) as model:
    print(model("What's the weather in Lagos?")["function_calls"])
    # [{'name': 'get_weather', 'arguments': {'city': 'Lagos'}}]
```

Or from the command line:

```sh
spindle run model.cact "What's the weather in Lagos?" --tools examples/tools.json
```

The tool part of the prompt is cached, so repeat requests with the same tools start faster. The attention
window is 1024 tokens by default; `Spindle(..., window=512)` or `--window 512` trades accuracy on long tool
lists for a little speed.

## Try it in the browser

The model also runs in WebAssembly. See [`demo/`](demo/) for a static page with tool calling, extraction,
sentiment and search-filter examples that you can host on GitHub Pages.

## Fine-tune

Training data is JSONL, one example per line (see `examples/train.jsonl`):

```json
{"query": "...", "tools": [...], "answers": [{"name": "...", "arguments": {...}}]}
```

```sh
spindle finetune --model model.cact --train train.jsonl --output runs/my-run
```

The result is `runs/my-run/model.cact`. Run `spindle finetune -h` for every setting.

Long runs can train in rounds. With `--updates-per-round 1000` the model and a `progress.json` are
written after every 1,000 updates, each round tokenizes only its own part of the data, and a run that
stopped goes on with `--resume runs/my-run`. `--max-minutes` stops before a round that would not fit.

Before training, some rows get a varied tool list, and copies of single-call rows without their
needed tool are added with "no call" as the answer. This teaches the model to check the tool list.
Tune it with `--tool-variety` and `--absent-tools` (0 turns each off).

## Extraction and classification

Both are tool calls. For extraction, give one `record_...` tool whose fields are what you want out of the
text. For classification, give one tool with an `enum` field. Pass `require_call=True` so the answer cannot
be empty:

```python
sentiment = [{"name": "classify_sentiment", "description": "Classify the sentiment of a text.",
              "parameters": {"type": "object", "required": ["sentiment"], "properties": {
                  "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]}}}}]
with Spindle("model.cact", sentiment) as model:
    model("The battery died after two days.", require_call=True)["function_calls"]
```

## Benchmarks

`spindle bench` scores a model on public test sets (needs `pip install -e '.[data]'`): Mobile Actions,
DroidCall and BFCL v4 for tool calling, When2Call and tool-removed requests for "no call", DSTC8 and SNIPS
for extraction, Rotten Tomatoes and DBpedia for classification. None of them is used for training.

```sh
spindle bench model.cact --quick                      # 200 cases per set
spindle bench model.cact --sets bfcl --window 1024    # one set, all cases
spindle bench qwen --server http://127.0.0.1:8080/v1  # any OpenAI-compatible server, same scoring
```

## The general training mix

`spindle build-data --output data/general` downloads open datasets and converts them to training rows:
function calling (xLAM, ToolACE, Glaive), requests that need no call (When2Call, out-of-scope requests),
extraction (MASSIVE, ATIS, Few-NERD, Pile-NER, generated records) and classification (SST-2, TweetEval,
emotion, AG News, Banking77). xLAM is gated: accept its terms on the Hub and run `huggingface-cli login`,
otherwise it is skipped.

## Layout

| Path | What it holds |
| --- | --- |
| `spindle/inference/` | Native engine binding, grammar-constrained decoding, the `Spindle` class |
| `spindle/model/` | The network in JAX/Flax, 4-bit fake quantization, runtime LoRA |
| `spindle/training/` | Data preparation, span attention supervision, the fine-tuning loop |
| `spindle/bench/` | Benchmark sets, scoring, and the runner behind `spindle bench` |
| `spindle/datasets/` | Converters for public datasets and the general training mix |
| `spindle/archive.py` | Reading and writing `.cact` model files |
| `native/` | The C++ CPU engine |

## License

Apache 2.0. See `LICENSE` and `NOTICE`.
