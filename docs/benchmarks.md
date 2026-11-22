# Benchmarks

Every model in this page is scored by the same code (`spindle bench`), on the same cases, with the same
rules. Nothing here is used for training: the training mix (`spindle build-data`) draws from other datasets.

## Results

<p align="center"><img src="images/benchmarks.svg" alt="Benchmark scores per model" width="760"></p>

| Benchmark | Metric | Spindle | Needle 3 | LFM3-1.7B |
| --- | --- | ---: | ---: | ---: |
| Mobile Actions | exact | 86.0 | 85.5 | **87.0** |
| DroidCall | exact | 33.5 | 41.0 | **50.0** |
| BFCL v4 | AST | 53.0 | 40.5 | **73.0** |
| When2Call | acc. | **70.0** | 64.5 | 68.0 |
| Tool removed | acc. | 53.5 | 38.0 | **65.5** |
| DSTC8 | F1 | 42.8 | 36.2 | **44.8** |
| SNIPS | F1 | 23.8 | 16.5 | **30.5** |
| SNIPS 7-way | F1 | 21.1 | 11.7 | **26.8** |
| Rotten Tomatoes | acc. | **65.0** | 48.5 | 15.5 |
| DBpedia | acc. | 46.5 | 31.0 | **79.5** |
| **Mean** | | 49.5 | 41.3 | **54.1** |
| Time per request (M1) | | 0.48 s | 0.38 s | 9.36 s |
| File size | | 75 MB | 35 MB | 1.1 GB |

<p align="center"><img src="images/size.svg" alt="Mean score against model size" width="760"></p>

Notes on these runs:

- LFM3-1.7B wrote text until the 512-token answer limit on most Rotten Tomatoes reviews instead of calling
  the tool, even though the request asked for a call (`tool_choice: required`). Its 15.5 there reflects that,
  not its sense of sentiment. It also makes its time per request higher.
- Spindle's enum debiasing is on in these runs (the default). Turned off (`--no-debias-enums`), DBpedia drops
  from 46.5 to 28.0 (mean 47.9); the other sets stay within 1.5 points.
- The per-set summaries, including call and decline rates, are in
  [`benchmarks/results/`](../benchmarks/results).

## The sets

| Set | Task | What a case is | Metric |
| --- | --- | --- | --- |
| [Mobile Actions](https://huggingface.co/datasets/google/mobile-actions) | Tool calling | A phone-assistant request with its tool list and a system line with the date | Exact match of every call and argument |
| [DroidCall](https://huggingface.co/datasets/mllmTeam/DroidCall) | Tool calling | An Android intent request | Exact match |
| [BFCL v4](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) | Tool calling + declining | Single-turn categories: simple, multiple, parallel, live, and the (live) irrelevance and relevance checks | BFCL's AST rule (any order, listed alternatives accepted) |
| [When2Call](https://huggingface.co/datasets/nvidia/When2Call) | Declining | A request where the right move is to call a tool, or not | Right decision (call / no call) |
| Tool removed | Declining | A single-call Mobile Actions case with the needed tool taken out of the list | Right decision (no call) |
| [DSTC8](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue) | Extraction | A user turn from the schema-guided dialogues; the service's intents are the tools | F1 over (tool, argument, value) |
| [SNIPS](https://github.com/MiuLab/SlotGated-SLU) | Extraction | A voice-assistant request; only the right intent is given as a tool | F1 over (tool, argument, value) |
| SNIPS 7-way | Extraction | The same requests with all seven intents as tools | F1 over (tool, argument, value) |
| [Rotten Tomatoes](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes) | Classification | A movie review; one tool with a `sentiment` enum | Accuracy |
| [DBpedia](https://huggingface.co/datasets/fancyzhx/dbpedia_14) | Classification | A short encyclopedia text; one tool with a 14-way `topic` enum | Accuracy |

Text values are compared without case and extra spaces, and `5` equals `5.0`. Everything else must match
exactly. Extraction and classification cases ask for a call (`require_call`), so a model cannot score by
saying nothing.

## How each model was run

All runs used one Apple M1 laptop (8 GB), one model at a time, 200 cases per set (a fixed random sample,
seed 0), greedy decoding.

| Model | File | Runtime | Settings |
| --- | --- | --- | --- |
| Spindle | `spindle.cact`, 4-bit | This repository's engine | Window 1024, output filters, copy guard and enum debiasing on (the defaults) |
| Needle 3 | The shipped 2-bit `needle3.cact` | The official Cactus engine | Its defaults |
| LFM3-1.7B | `LFM3-1.7B-Q4_K_M.gguf` | llama.cpp `llama-server` (b11960) | Chat template with tools, thinking off, 4 threads |

Any model behind an OpenAI-compatible server can be scored the same way:

```sh
llama-server -m model.gguf --jinja --port 8080
spindle bench my-model --server http://127.0.0.1:8080/v1 --quick --output report.json
```

## Reproduce

```sh
pip install -e '.[data]'
spindle bench models/spindle.cact --quick --output report.json   # about 30 minutes on an M1
```

`--quick` takes 200 cases per set; leave it out for every case, or use `--sets` and `--limit` for a part. The
report has every case with the model's output, so errors are easy to read through.

To redraw the charts, put the report (with `name`, `size_mb` and `order` added) in `benchmarks/results/`
and run `python benchmarks/plot.py`.

## Reading the numbers

- With 200 cases a score moves by about ±3 points from the sample alone (one standard error near 50%), so
  differences of a few points are not meaningful.
- Spindle checks each call against the request before returning it (`filters=True`, see
  [what changed from Needle 3](changes-from-needle-3.md)). This is what makes it decline when no tool fits; with the filters off
  it calls a tool far more often, also when it should not.
- Speed is the average time per request on the same laptop, including the prompt. llama.cpp uses the GPU on
  Apple silicon; the two Needle-based engines use the CPU only.
