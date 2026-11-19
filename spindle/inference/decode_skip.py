"""Pick the layers that decode skip should skip, for any model.

Decode skip lets generated tokens skip some layers (Engine.set_decode_skip). Which layers a model
can do without differs from model to model, so this measures it instead of assuming it:

  1. The full model (nothing skipped) answers a set of requests. Its answers are the reference,
     so no labels are needed.
  2. Each candidate list replays those answers token by token with its layers skipped. Its cost is
     how much less likely the full model's own answer tokens become (extra loss per token, in nats).
  3. Layers are added one at a time, always the one that costs least next to those already chosen.

The prompt always runs every layer, and the prefix cache keeps it, so each candidate only pays for
its answer tokens. Afterwards, fine-tune with `--skip-layers` so the model learns to work without
those layers (this heals most of the loss) and run with `--decode-skip`.
"""
import time

import numpy as np

from spindle import Spindle


def format_layers(layers):
    """(5, 6, 7, 8, 10) -> '5-8,10'."""
    parts, layers = [], sorted(layers)
    for layer in layers:
        if parts and layer == parts[-1][1] + 1:
            parts[-1][1] = layer
        else:
            parts.append([layer, layer])
    return ','.join(str(a) if a == b else f'{a}-{b}' for a, b in parts)


def references(model, cases, max_new_tokens=256):
    """The full model's answers: (prompt ids, answer ids) per case."""
    engine, inner, fed = model.engine, model.engine.eval, []

    def record(tokens):                     # the prompt comes as one call, then one call per answer token
        fed.append([int(t) for t in tokens])
        return inner(tokens)

    engine.set_decode_skip(None)
    engine.eval = record
    result, started = [], time.time()
    try:
        for i, case in enumerate(cases, 1):
            fed.clear()
            model.set_tools(case['tools'], system=case.get('system', ''))
            model(case['query'], max_new_tokens=max_new_tokens, require_call=case.get('require_call', False))
            # Fed tokens predict the next one; the final token is never fed back, so it is not scored.
            result.append((fed[0], [token for (token,) in fed[1:]]))
            if i % 10 == 0 or i == len(cases):
                print(f'  reference answers {i}/{len(cases)}  ({time.time() - started:.0f}s)', flush=True)
    finally:
        del engine.eval                     # back to the class method
    return [(prompt, answer) for prompt, answer in result if answer]


def answer_losses(engine, refs, candidates):
    """Mean loss per answer token of the full model's answers, for each candidate skip list."""
    totals, count = np.zeros(len(candidates)), 0
    for prompt, answer in refs:
        engine.set_prefix(len(prompt) - 1)          # every candidate after the first reuses the prompt
        for j, layers in enumerate(candidates):
            engine.set_decode_skip(layers)
            engine.reset()
            logits = engine.eval(prompt)
            for k, token in enumerate(answer):
                top = logits.max()
                totals[j] += top + np.log(np.exp(logits - top).sum()) - logits[token]
                if k + 1 < len(answer):
                    logits = engine.eval([token])
        count += len(answer)
    engine.set_decode_skip(None)
    return totals / count


def pick_layers(model, cases, count, pool=6, max_new_tokens=256):
    """Greedy choice of `count` layers. Returns [(layers, extra loss per token)] after each step.
    After the first step, only the `pool` layers that were cheapest on their own are tried."""
    engine, started = model.engine, time.time()
    refs = references(model, cases, max_new_tokens)
    tokens = sum(len(answer) for _, answer in refs)
    layers = engine.geometry[10]
    print(f'Scoring every layer on its own ({len(refs)} answers, {tokens} tokens) ...', flush=True)
    scores = answer_losses(engine, refs, [()] + [(layer,) for layer in range(layers - 1)])
    base, alone = scores[0], scores[1:] - scores[0]
    for layer in range(layers - 1):
        print(f'  layer {layer:2d}  extra loss {alone[layer]:7.4f}', flush=True)
    print(f'  layer {layers - 1:2d}  (the read-out layer always runs)', flush=True)
    order = [int(layer) for layer in np.argsort(alone)]
    chosen, steps = [order[0]], [((order[0],), float(alone[order[0]]))]
    print(f'step 1: skip {format_layers(chosen):18s} extra loss {steps[-1][1]:.4f}', flush=True)
    while len(chosen) < count:
        options = [layer for layer in order if layer not in chosen][:pool]
        trials = [tuple(sorted(chosen + [layer])) for layer in options]
        costs = answer_losses(engine, refs, trials) - base
        best = int(np.argmin(costs))
        chosen.append(options[best])
        steps.append((trials[best], float(costs[best])))
        print(f'step {len(chosen)}: skip {format_layers(chosen):18s} extra loss {costs[best]:.4f}  '
              f'({time.time() - started:.0f}s)', flush=True)
    return steps


def load_cases(data=None, per_set=6):
    """Requests from a JSONL file ({query, tools, [system]} per line) or a few from each benchmark set."""
    if data:
        import json
        with open(data, encoding='utf-8') as handle:
            return [json.loads(line) for line in handle if line.strip()]
    from spindle.bench import sets
    return [case for name in ('mobile_actions', 'droidcall', 'dstc8', 'bfcl', 'when2call')
            for case in sets.load(name, per_set)]


def main(model_path, *, count=7, data=None, per_set=6, pool=6, window=0, threads=0, library=None):
    cases = load_cases(data, per_set)
    with Spindle(model_path, window=window, threads=threads, library=library) as model:
        steps = pick_layers(model, cases, count, pool)
    layers, cost = steps[-1]
    total = model.engine.geometry[10]
    print(f'\nSuggested: --skip-layers {format_layers(layers)}  (generated tokens run {total - len(layers)} '
          f'of {total} layers; extra loss {cost:.4f} per token before fine-tuning)')
    print('Pick fewer layers if a step above jumped in cost. Then fine-tune with --skip-layers to heal the loss,')
    print('check the score with `spindle bench --decode-skip`, and run with --decode-skip.')
    return steps
