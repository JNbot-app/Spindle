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

## Fine-tune

Training data is JSONL, one example per line (see `examples/train.jsonl`):

```json
{"query": "...", "tools": [...], "answers": [{"name": "...", "arguments": {...}}]}
```

```sh
spindle finetune --model model.cact --train train.jsonl --output runs/my-run
```

The result is `runs/my-run/model.cact`. Run `spindle finetune -h` for every setting.

## Layout

| Path | What it holds |
| --- | --- |
| `spindle/inference/` | Native engine binding, grammar-constrained decoding, the `Spindle` class |
| `spindle/model/` | The network in JAX/Flax, 4-bit fake quantization, runtime LoRA |
| `spindle/training/` | Data preparation, span attention supervision, the fine-tuning loop |
| `spindle/archive.py` | Reading and writing `.cact` model files |
| `native/` | The C++ CPU engine |

## License

Apache 2.0. See `LICENSE` and `NOTICE`.
