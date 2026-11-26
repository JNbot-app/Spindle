# What changed from Needle 3

Spindle starts from [Needle 3](https://github.com/cactus-compute/needle) by Cactus Compute, a 121M-parameter Simple Attention Network for on-device tool calling. The core architecture stays mostly the same. The changes are in the MLP blocks, how the model is fine-tuned, and the engine that runs it.

| Area | Needle 3 | Spindle |
| --- | --- | --- |
| MLP mixing | Kronecker: one 32x32 matrix shared by all 32 groups | Monarch: one 32x32 matrix per group |
| Fine-tuning | LoRA on attention, merged into 4-bit weights at export | Full fine-tune, separate rate for MLP blocks, unmerged FP16 LoRA |
| Training loss | Cross-entropy on the answer | Weighted argument and no-call loss, plus span attention loss |
| Training data | Used as given | Varied tool lists, missing-tool negatives, and two-call queries |
| Call or no call | Learned from answers | Also distilled from a reasoning-trained teacher |
| Attention window | 256 tokens in file header | 1024 tokens by default, configurable at runtime (`window`) |
| Output | Reasoning, then calls | Calls by default, optional reasoning note |
| Confidence | Learned confidence head | Per-call token probability measured during decoding |
| Decoding | Grammar-constrained | Same, plus copy guard and enum debiasing (both on by default) |
| Engine | Cactus prebuilt engines | Custom C++17 CPU engine for the new tensors |

Unless noted otherwise, the ablation numbers below come from development runs on earlier checkpoints against matched controls. For the released model against other models, see [benchmarks](benchmarks.md).

## Why 4-bit weights do not move

Needle stores almost all weights as 4-bit codes in groups of 128 sharing one FP16 scale. Training uses a straight-through estimator [4]: the forward pass uses rounded weights, while gradients update an FP32 copy. A 4-bit code only flips when the FP32 copy crosses a quantization boundary.

In practice, it almost never does. After full fine-tunes at learning rates of `1e-5` and even `5e-5`, zero 4-bit codes changed. The nearest quantization boundary was 11 to 460 times farther away than Adam moved any weight during a run. Merged LoRA hits the same wall: in one test, merging changed only 1,100 out of 35.9M codes, wiping out a +0.56 nat teacher-forced gain (the exported model scored -0.03).

Cranking the learning rate high enough to flip codes destabilizes the rest of the network. Spindle instead puts trainable capacity into FP16 tensors where small gradient steps actually persist.

## Architecture changes

### 1. Monarch mixing in the MLP

Needle's Hadamard MLP mixes a 1024-wide vector in three stages, applying a pair of 32x32 matrices to 32 groups at each stage. Needle 3 calls this "Monarch", but every group shares the same matrix (the Kronecker special case). A full Monarch matrix [1] gives each group its own 32x32 matrix.

Because each token still performs 32 small 32x32 products per stage, the multiply count stays identical. Parameter count per layer grows from 25,600 to 216,064 (+3.8M total, or +7.6 MB in FP16, about 3% of the file). These weights are stored in FP16 so fine-tuning updates them directly. Copying the shared Needle 3 matrix into every group (`spindle monarch`) initializes the block identically with zero loss at conversion time.

In a controlled test where only the 20 MLP blocks were trained for 1,000 steps:

| Block | Trainable weights | Extra compute | Held-out NLL |
| --- | ---: | ---: | ---: |
| Needle 3 (Kronecker) | 512k | 0 | 0.024 |
| Kronecker + low-rank path | 4.3M | ~2x block | 0.018 |
| **Monarch** | 4.3M | 0 | **0.014** |

Monarch beats an equal-parameter low-rank control, showing the gain comes from the structure rather than size alone. The same test showed the MLP blocks learn best at around `1e-3` while the backbone uses `1e-5`, so `hadamard_lr` defaults to `1e-3`.

### 2. Runtime LoRA on attention

Because the 4-bit attention weights stay effectively frozen, Spindle adds an unmerged rank-16 LoRA [2] to the `q`, `k`, `v`, gate, and output projections across all 20 layers (2.0M parameters, 4 MB in FP16). Following QLoRA [3], the LoRA weights live next to the quantized base weights and the engine adds `x * A^T * B` at runtime:

- Training sees the exact same `base + A * B` computation as inference.
- `lora_lr` defaults to `3e-3`. At `1e-3` the update stayed around 0.6% of the weight norm, while `3e-3` reached about 4%.
- `B` starts at zero so a fresh LoRA leaves outputs unchanged initially. When fine-tuning a model that already has a LoRA, training resumes from it.
- Runtime cost is roughly +0.1 ms per token on an Apple M1 (3 to 7%).

On its own, the LoRA only gave a small improvement, but it unlocks attention updates needed for span supervision below.

### 3. Wider attention window

Needle 3 caps decoder attention at the last 256 tokens. Since tool definitions sit at the start of the prompt, the right tool was often 300 to 500 tokens before the answer and out of sight. In those cases, Needle 3 got only 3 of 49 answers right.

Raising the window to 1024 without retraining produced the single largest accuracy jump. On prompts longer than 512 tokens with three extra tools, a 512-token window got 17 of 74 right while 1024 got 53 of 74. Since the KV cache is 8-bit, the extra memory footprint is small. Spindle defaults to 1024 and exposes `window` at runtime (`Spindle(..., window=512)` or `--window 512`).

## Training and data changes

### 4. Span attention loss

Many remaining errors were copy mistakes: picking a number instead of a name, grabbing a pronoun, pulling from the wrong clause, or truncating a value. When an argument value appears in the query, we know which input tokens the model should attend to, similar to supervised attention [5] and copy mechanisms [6].

For every string or number argument found in the query (up to 12 targets per row, covering both required and optional arguments), `spindle finetune` supervises two answer positions found by perturbing the key or value:

1. the position that decides to write the argument's key
2. the position that writes the value's first token

Across layers 10 to 19, at least one attention head per layer should focus on the value's query span. Each layer's loss is the best head's `-log(attention mass on span)`, averaged over layers and scaled by `span_weight` (default `0.5`). Taking the best head leaves the model free to assign copying to whichever head it chooses without forcing all heads together. This loss is training-only and relies on the runtime LoRA so attention weights can adapt.

In matched comparisons, it eliminated most targeted copy errors (wrong spans, cross-clause values, truncated names).

### 5. Loss weights

Most tokens in a JSON answer are boilerplate syntax and tool names, while a "no call" answer is only a few tokens long. Plain cross-entropy barely weights either. Spindle scales:

- tokens inside argument values by 4x (`argument_weight`)
- every token in a no-call row by 20x (`no_call_weight`)

The batch loss is normalized by total token weight, shifting focus without changing the effective learning rate.

### 6. Tool-list variety and missing-tool negatives

In many datasets, the tool list hardly changes for a given query type, so models memorize query-to-tool shortcuts instead of reading the schemas. Adding three unrelated tools dropped accuracy by about 7%, and removing the needed tool still triggered a call 60% of the time.

Before training, `spindle finetune` modifies training rows (leaving validation rows untouched) in `spindle/training/tool_lists.py`:

- **Varied lists (`tool_variety`, default `0.5`)**: Keeps required tools, drops some unused ones, adds 1 to 3 tools sampled from other rows, and shuffles the order. Tools sharing a name word or synonym with the target tool (such as `send_message` and `send_email`) are never added as distractors.
- **Missing-tool negatives (`absent_tools`, default `0.02` times row count)**: Copies single-call rows, strips out the needed tool, varies the remaining list, and sets the answer to no call.

This raised the decline rate when the needed tool is absent from 39% to 94% (varied lists alone helped little without missing-tool negatives). The trade-off is higher sensitivity to unfamiliar phrasings, so provide varied request phrasings per tool, or lower `absent_tools` (down to `0`) if missed calls are more costly than false triggers.

### 7. Distilling call decisions from a reasoning teacher

Models trained to write a short reasoning note before calling tools are better at deciding when to abstain (declining 66% of missing-tool cases vs 45% without notes), but generating reasoning tokens slows down CPU inference and hurt other metrics.

Spindle keeps only the decision boundary (`spindle finetune --teacher teacher.cact --distill-weight 1`, in `spindle/training/decision.py`). During fine-tuning, a reasoning-trained teacher evaluates the same prompt and its decline probability serves as a soft target for the single token right after `<tool_call>` (`[]` for no call vs `[{"` for a call). Nothing else from the teacher is copied, adding roughly 12% to training time.

On matched 200-case runs, distillation improved When2Call from 60.5 to 68.0, tool-removed accuracy from 45.5 to 48.0, and the 10-benchmark mean from 45.2 to 46.5.

### 8. Two-call combined requests

Because most public rows expect a single call, the model tended to stop after one call on multi-part queries. `spindle build-data --combined 0.15` joins pairs of single-call rows with `" and "` and combines their answers. Training with ~15% combined rows raised the tool-removed score from 46.0 to 53.5 and the 10-set mean from 48.6 to 49.5 (Mobile Actions shifted from 88 to 86, within noise).

## Decoding and inference engine

### 9. Custom C++17 engine

Needle 3's prebuilt engines expect shared 2D mixing matrices and have no runtime LoRA support, so Spindle ships a ~4,000-line C++17 engine in `native/`:

- **In-place loading**: Memory-maps `.cact` files and reads Monarch factors and LoRA weights directly in FP16. Also opens Needle 3 `.cact` files (2-bit weights): 2-bit codes are read in place by native 2-bit kernels on ARM (other CPUs widen a few rows at a time), so they need half the memory of 4-bit.
- **Quantized kernels**: Implements 4-bit kernels for ARM NEON (with and without dotprod), AVX2, SSE2, and scalar C++, using Accelerate on Apple and OpenBLAS elsewhere for prefill.
- **Training parity**: Runs 8-bit activations and an 8-bit KV cache matching the simulated quantization in JAX. Across full prompts and answers, engine predictions match JAX token for token.
- **Prompt formatting and normalization**: Keeps tool schemas as written (converting tool names to `snake_case`), normalizes system timestamps into the short training format (`date: 2024-06-06 Thu 09:58`), and restores exact input spelling when a text argument differs from a query span only in case or punctuation (`03 1234-5678` becomes `(03) 1234-5678`).
- **Tool-prefix caching**: Caches KV state immediately after the tool definitions and always aligns prefill chunks to end at the tool boundary. Cache hits produce bit-for-bit identical results to cold runs while skipping tool prefill (`metrics['cached_prompt_tokens']`).
- **Threading**: Prefill uses half the performance cores (capped at 4), with output independent of thread count.

Monarch mixing adds about 15% to decode latency per token compared to shared matrices due to less regular memory access.

### 10. Calls by default and per-call probability

Spindle generates calls directly by default. Reasoning notes can still be enabled (`reasoning=True` or `spindle build-data --reasoning`), though forcing notes reduced call accuracy on CPU benchmarks.

Needle 3 uses a learned confidence head, which drifts out of calibration after fine-tuning. Spindle replaces it with the mean token probability of each decoded call, renormalized over grammar-valid tokens. This requires no training, scores each call individually, and achieved 0.88 AUC separating correct calls from incorrect ones. Calls below `min_call_probability` are filtered out alongside Needle-style grounding, negation, and duplicate checks.

### 11. Copy guard and enum debiasing

Both rules run token by token right after grammar constraints [7] inside `native/grammar.cpp`, so Python and the browser WebAssembly demo behave identically:

- **Copy guard (`copy_margin=4`, `0` disables)**: Small models sometimes start copying a phrase from the request and paraphrase halfway through (`"Café Grano de Oro at 25 Calle del Prado"` becomes `"... Oro 25 Calle del Prado"`). If the string generated so far matches a substring of the query and the top token diverges from it, the copy guard substitutes the token that continues the query match if it falls within `4` in log probability. Ending the string is never blocked, and values absent from the query (such as dates or codes) are untouched. This raised Mobile Actions from 76 to 82 with no regressions elsewhere.
- **Enum debiasing (`debias_enums=True`, disable via `debias_enums=False` or `--no-debias-enums`)**: On large closed label sets (like 14-way topic classification), small models skew toward a handful of labels regardless of input. After 8 requests sharing the same field and enum values, this tracks running mean label probabilities and picks by `p(label) / mean p(label)`. It requires no ground-truth labels, assuming labels appear roughly evenly over time (disable it if one label dominates your traffic). On DBpedia, it raised accuracy from 28.0 to 46.5.

### 12. Decode skip

Each generated token normally passes through all 20 layers. Decode skip lets generation skip a specified subset of layers while still running all 20 layers on the prompt (and never skipping the final readout layer). It is off by default and toggleable per call, even between cached requests (`Spindle(..., decode_skip=True)`, `--decode-skip`, `decode_skip='5-8,10'`, or `model.engine.set_decode_skip(None)`).

To avoid guessing which layers matter, `spindle pick-skip model.cact` (with optional `--data` for task queries) generates answers with the full model, replays them token by token while dropping candidate layers, and greedily builds the lowest-cost skip list in a few minutes with no labels needed.

`spindle finetune --skip-layers ...` then heals the remaining layers: skipped answer tokens keep their residual stream unchanged, skip writing KV entries, and zero out their `q`/`k`/`v` taps. The skip mask is saved in header fields 45 and 46 of the `.cact` file and preserved in subsequent fine-tunes (`--skip-layers none` clears it).

On an earlier checkpoint healed for 1,000 updates with layers `5, 6, 7, 8, 10, 13, 15` skipped (evaluated across 400 free-generation cases on an Apple M1, 80 each from Mobile Actions, DroidCall, DSTC8, BFCL, and When2Call):

| Model | Mean | Mobile Actions | DroidCall | DSTC8 | BFCL | When2Call | Decode ms / token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Checkpoint, all layers | 53.7 | 78.8 | 36.2 | 43.7 | 48.8 | 61.2 | 2.93 ms |
| Healed copy, decode skip on | 52.1 | 71.2 | 31.2 | 41.6 | 47.5 | 68.8 | 2.11 ms (1.39x) |

Both models agreed on 339 of 400 cases, with 27 fixed and 34 broken (McNemar `p = 0.44`). Naive skipping of every other middle layer (`4, 6, ..., 16`) lost roughly 9 points, showing that model-specific layer selection matters. No decode-skip checkpoint ships with the current release, so run `pick-skip` and a short `--skip-layers` fine-tune on the released weights if you want one.

## Costs at a glance

| Change | Size | Decode speed |
| --- | --- | --- |
| Monarch factors | +7.6 MB | ~15% slower |
| Runtime LoRA | +4 MB | 3 to 7% slower |
| 1024-token window | 4x KV cache (8-bit) | Slight prefill cost on long prompts |
| Tool-prefix cache | One copy of cached KV state | Skips tool prefill on repeat requests |
| Decode skip (off by default, needs healed weights) | None | 1.39x faster per generated token (~1.6 pts drop on earlier checkpoint) |
| Span loss, loss weights | None | None |
| Teacher distillation, two-call rows | None | None (~12% slower training) |
| Copy guard, enum debiasing | None | Negligible |

## References

1. T. Dao et al. *Monarch: Expressive Structured Matrices for Efficient and Accurate Training.* ICML 2022. [arXiv:2204.00595](https://arxiv.org/abs/2204.00595)
2. E. J. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
3. T. Dettmers et al. *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
4. Y. Bengio, N. Léonard, A. Courville. *Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation.* 2013. [arXiv:1308.3432](https://arxiv.org/abs/1308.3432)
5. L. Liu et al. *Neural Machine Translation with Supervised Attention.* COLING 2016. [arXiv:1609.04186](https://arxiv.org/abs/1609.04186)
6. J. Gu et al. *Incorporating Copying Mechanism in Sequence-to-Sequence Learning.* ACL 2016. [arXiv:1603.06393](https://arxiv.org/abs/1603.06393)
7. S. Geng et al. *Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning.* EMNLP 2023. [arXiv:2305.13971](https://arxiv.org/abs/2305.13971)
8. Cactus Compute. *Needle 3.* [github.com/cactus-compute/needle](https://github.com/cactus-compute/needle)