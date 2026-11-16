# What changed from Needle 3

Spindle starts from [Needle 3](https://github.com/cactus-compute/needle) by Cactus Compute, a 121M-parameter
Simple Attention Network for on-device tool calling. The architecture stays almost the same. The changes are
in three places: the MLP block, how the model is fine-tuned, and the engine that runs it.

| Area | Needle 3 | Spindle |
| --- | --- | --- |
| MLP mixing | Kronecker: one 32×32 matrix shared by all 32 groups | Monarch: one 32×32 matrix per group |
| Fine-tuning | LoRA on attention, merged into the 4-bit weights at export | Full fine-tune, with a separate rate for the MLP blocks and an unmerged FP16 LoRA |
| Training loss | Cross-entropy on the answer | Same, plus weights on argument values and no-call answers, plus a span attention loss |
| Training data | Used as given | Varied tool lists, plus "no call" rows where the needed tool is missing |
| Attention window | 256 tokens | 512 tokens, can be raised at run time (`window`) |
| Output | Reasoning, then calls | Calls only |
| Confidence | Learned confidence head | Probability of each call, measured during decoding |
| Engine | Cactus prebuilt engines | Own C++ CPU engine that reads the new tensors |

The numbers below come from development experiments on earlier checkpoints, each against a matched control.
Full benchmarks will be added in a separate document.

## The finding behind most changes: 4-bit weights do not move

Needle stores almost all weights as 4-bit codes, one per weight, in groups of 128 that share one FP16 scale.
Training uses a straight-through estimator [4]: the forward pass sees the rounded weights, and the gradient
updates a full-precision copy. A code only changes when that copy crosses the boundary to the next 4-bit
level.

In practice, it never does. After full fine-tunes at learning rate 1e-5, and even at 5e-5, **not one 4-bit code
had changed**. The nearest boundary was 11 to 460 times farther away than Adam moved the weight during a run.
Merged LoRA has the same problem: in one run, export changed only 1,100 of 35.9M codes. A teacher-forced check
showed the LoRA had learned (+0.56 nats on its targets), but the exported model kept none of it (−0.03).

Raising the learning rate until codes flip would disturb the whole model. So Spindle moves the learning into
places where small updates survive: the FP16 tensors. The next two changes do exactly that.

## 1. Monarch mixing in the MLP

Needle's MLP block is a Hadamard MLP. It mixes a 1024-wide vector in three stages, and each stage applies a
pair of 32×32 matrices to 32 groups. Needle 3 calls this "Monarch", but every group shares the same matrix.
That is the Kronecker special case. A full Monarch matrix [1] gives each group its own matrix.

**Why it makes sense**

- **Same compute.** Each token still does 32 small 32×32 products per stage, so the multiply count does not
  change.
- **Much more freedom.** Weights per layer grow from 25,600 to 216,064. That is 3.8M more in total, or 7.6 MB in
  FP16 (about 3% of the file).
- **The new weights are FP16.** They are not 4-bit, so fine-tuning really changes them.
- **It starts exactly where Needle 3 is.** Copying the shared matrix into every group gives the same function,
  so the conversion (`spindle monarch`) loses nothing.

**Evidence.** In a controlled test, the whole model was frozen and only the 20 MLP blocks were trained (1,000
steps). Held-out loss:

| Block | Trainable weights | Extra compute | Held-out NLL |
| --- | ---: | ---: | ---: |
| Needle 3 (Kronecker) | 512k | 0 | 0.024 |
| Kronecker + low-rank path of the same size | 4.3M | about 2× block | 0.018 |
| **Monarch** | 4.3M | 0 | **0.014** |

Monarch beats a control with the same number of weights, so the gain comes from the structure, not only the
size. Synthetic tests (memorisation, XOR pairs, routing) pointed the same way.

**Its own learning rate.** The same test showed that the MLP blocks learn best at about 1e-3, while the
backbone uses 1e-5. Spindle gives the Hadamard blocks their own rate (`hadamard_lr`, default 1e-3). In full
fine-tunes this helped a lot by itself, and Monarch added a clear gain on top.

## 2. Runtime LoRA on attention

Attention weights are 4-bit, so they stay frozen in practice (see above). Spindle adds a rank-16 LoRA [2] to
the q, k, v, gate and output projections of all 20 layers (2.0M parameters, 4 MB in FP16). Unlike Needle 3's
LoRA, it is **not merged** into the 4-bit weights. It is stored next to them, and the engine adds `x·Aᵀ·B` to
each projection at run time. The idea is the same as QLoRA [3]: keep the quantized base, and train a small
update in higher precision.

- Training sees exactly what the engine computes: the 4-bit weights plus `A·B`.
- The LoRA uses its own rate (`lora_lr`, default 3e-3). At 1e-3 the update stayed at about 0.6% of the weight
  size, while at 3e-3 it reached about 4%.
- B starts at zero, so a new LoRA does not change the model at the start. When the parent model already has a
  LoRA, training continues it.
- Cost: about +0.1 ms per token on an Apple M1 (3-7%).

On its own, the LoRA gave only a small gain, because the training data had little left to teach. Its real role
is to let the attention learn at all, and that matters for the next change.

## 3. Span attention loss

Most remaining errors were copying the wrong part of the query into an argument: a number instead of a name,
a pronoun, a value from the next clause, or a value cut short. When an argument value appears in the query,
we know exactly which query tokens it should come from. Spindle uses that as an extra training signal,
similar in spirit to supervised attention [5] and copy mechanisms [6].

**How it works**

- **Targets.** For every string or number argument that appears in the query, two answer positions are
  supervised:
  1. the position that decides to write the argument's key,
  2. the position that writes the value's first token.

  Each position is found by changing the key or the value in the answer and looking for the first token that
  changes. Up to 12 targets are used per row.
- **Loss.** In each of layers 10-19, at least one head should put its attention on the value's span in the
  query. Per layer, the loss is the best head's −log(attention mass on the span), averaged over the layers and
  weighted by `span_weight` (default 0.5).
- **Why "best head".** It leaves the model free to choose which head does the copying, and it does not force
  every head to look at the same place.
- **Training only.** The loss changes nothing at inference. Attention can only change through the runtime
  LoRA, which is why both are always on together.

**What it fixed.** In a matched comparison, it removed most of the copy errors it targeted: wrong spans, values
from the wrong clause, and cut-off names. A short run left out more arguments, but that went away with full
training.

The released model was trained with the first version of the targets: value positions of required string
arguments only. `spindle finetune` uses the extended version, which covers numbers, optional arguments and the
key position, because left-out optional arguments were the next most common error.

## 4. Loss weights

Most answer tokens are fixed JSON syntax and tool names. The tokens that decide whether an answer is right
are the argument values. A "no call" answer is only a few tokens long, so plain cross-entropy barely trains
that decision. Spindle weights both:

- tokens inside argument values ×4 (`argument_weight`),
- every token of a row without a call ×20 (`no_call_weight`).

The loss is normalised by the total weight of the batch, so these weights change the balance between parts of
the answer, not the overall learning rate.

## 4b. Tool-list variety

In most tool-calling data the tool list is almost fixed for each kind of request. The model can then
answer from the request alone and never learns to read the list. In our tests, three unrelated tools added
to a request cost about 7% of the right answers, and with the needed tool removed the model still called a
tool 60% of the time.

`spindle finetune` therefore edits the training rows (not the validation rows) before training. Both edits
keep the request text (`spindle/training/tool_lists.py`):

- **Varied lists** (`tool_variety`, default 0.5): a call row keeps its needed tools, drops some unused ones,
  gets 1-3 tools from other rows and a new order. The answer stays the same.
- **Tool-absent rows** (`absent_tools`, default 0.02 × the row count): a copy of a single-call row without
  its needed tool and with a varied list. The answer is "no call".

Every added tool is a right answer in other rows, so an unfamiliar tool or an unusual list says nothing about
the answer. Tools that could do the same job as the needed tool (shared name word or a synonym, e.g.
`send_message` / `send_email`) are never added next to it.

**Effect.** With the needed tool missing, the model declined 94% of the time (39% before). Varied lists
alone changed little; the tool-absent rows do the work.

**Cost.** The model also declines more real requests whose wording a tool never had in training. Give each
tool several request wordings in your data. If missing a call costs you more than a wrong call, lower
`absent_tools` or set it to 0.

## 5. A 512-token attention window

Needle 3 limits decoder attention to the last 256 tokens. Tool definitions come first in the prompt, so with
several tools the right definition was often 300-500 tokens before the answer, and the model could not see it.
It then picked a visible tool near the end of the list, or no tool. In those cases only 3 of 49 answers were
right.

Raising the window to 1024, with no retraining, gave the largest single gain we found. 512 gave the same gain
at lower latency, and training with the same window added a little more. The KV cache is 8-bit, so the extra memory
is small.

With five or six tools, prompts pass 512 tokens again and the needed tool falls out of view. In our tests,
on prompts longer than 512 tokens with three extra tools, 512 kept 17 of 74 right answers and 1024 kept 53,
with no retraining. So Spindle runs with a window of 1024 by default (the local attention width the model
was built with), and the window can be set at run time: `Spindle(..., window=512)` or `--window 512`.

## 6. Calls only, and a per-call probability

**Reasoning is optional.** Needle 3 writes a short reasoning note before its calls. Spindle lets a model do that when it prefers to (`reasoning=True`) and can train with notes built from the answers (`spindle build-data --reasoning`); rows without a note train calls only.
Reasoning costs extra tokens on a CPU, and in our tests forcing it lowered call accuracy.

**Per-call probability.** Needle 3 reports a confidence from a learned head, and a local fine-tune leaves that
head uncalibrated. Spindle instead measures each call while decoding: the mean probability of its tokens,
renormalised over the tokens the grammar allows. This needs no training, so it stays valid after every
fine-tune, and it scores each call separately. In our tests it told right calls from wrong ones with an AUC of
0.88. Calls below `min_call_probability` are held back.

The output filters (grounding, negation, duplicates) follow Needle's output contract, rewritten without
domain-specific rules.

## 7. Inference engine

Needle 3's prebuilt engines expect shared 2D mixing matrices and have no LoRA block, so they cannot run this
model. Spindle has its own small engine (C++17, about 3,300 lines, in `native/`):

- **Reads the file in place.** It maps the `.cact` file into memory. The Monarch factors and the LoRA are read
  as FP16 straight from the file.
- **Fast 4-bit kernels** for ARM NEON (with and without dot-product), AVX2, SSE2 and plain C++. Prefill uses
  Accelerate on Apple and OpenBLAS elsewhere.
- **Same rounding as training.** It uses 8-bit activations and an 8-bit KV cache, the same rounding the
  training graph simulates. Across a full prompt and answer, its predicted token matched the JAX model at every
  position.
- **Grammar-constrained decoding** [7] from the tools' JSON Schemas, so every answer parses and every call
  fits its schema. The same pass computes the per-call probability.
- **Tool-prefix cache.** The tool list comes first in the prompt and is the same for every request with the
  same tools. The engine keeps its state after the tool part (whole 128-token prefill chunks) and reuses it
  when the next prompt starts with the same tokens. The output is bit-for-bit the same; only the prefill of
  the tool list is skipped. `metrics['cached_prompt_tokens']` shows how much was reused.

- **Reads Needle 3 files too.** Weights stored with fewer than 4 bits (the shipped Needle 3 model uses
  2) are widened to the 4-bit layout when the file is loaded, so they use the same kernels.
- **The prompt the model was trained on.** Tools are shown as written (only names become snake_case), and
  a timestamp in the system prompt becomes the short date fact used in training (`date: 2024-06-06 Thu 09:58`).
- **Exact copies.** A text argument that differs from a passage of the request only in punctuation or case
  takes the request's spelling (`03 1234-5678` becomes `(03) 1234-5678`).
- **Small worker team.** Prefill runs its per-token loops on a few threads (half the performance cores, at
  most four); results do not depend on the thread count.

Monarch costs about 15% more decode time per token than shared matrices. Its arithmetic is the same, but the
memory access is less regular.

## Costs at a glance

| Change | Size | Decode speed |
| --- | --- | --- |
| Monarch factors | +7.6 MB | about 15% slower |
| Runtime LoRA | +4 MB | 3-7% slower |
| 512-token window | KV cache ×2 (8-bit) | longer prompts cost a bit more |
| Tool-prefix cache | one copy of the cached state | prefill of the tool list skipped on repeat requests |
| Span loss, loss weights | none | none |

## References

1. T. Dao et al. *Monarch: Expressive Structured Matrices for Efficient and Accurate Training.* ICML 2022.
   [arXiv:2204.00595](https://arxiv.org/abs/2204.00595)
2. E. J. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
   [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
3. T. Dettmers et al. *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023.
   [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
4. Y. Bengio, N. Léonard, A. Courville. *Estimating or Propagating Gradients Through Stochastic Neurons for
   Conditional Computation.* 2013. [arXiv:1308.3432](https://arxiv.org/abs/1308.3432)
5. L. Liu et al. *Neural Machine Translation with Supervised Attention.* COLING 2016.
   [arXiv:1609.04186](https://arxiv.org/abs/1609.04186)
6. J. Gu et al. *Incorporating Copying Mechanism in Sequence-to-Sequence Learning.* ACL 2016.
   [arXiv:1603.06393](https://arxiv.org/abs/1603.06393)
7. S. Geng et al. *Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning.* EMNLP 2023.
   [arXiv:2305.13971](https://arxiv.org/abs/2305.13971)
8. Cactus Compute. *Needle 3.* [github.com/cactus-compute/needle](https://github.com/cactus-compute/needle)
