"""Build the general training mix from public datasets.

    spindle build-data --output data/general --rows 128000

Writes train.jsonl, validation.jsonl (the same number of held-out rows per task) and
summary.json. Rows keep a `task` tag (tools, no_call, extract, classify). The training file is
ordered so that every stretch of it has the same task shares: a trainer that reads it in chunks
sees the same mix in every chunk.
"""
import json
import random
from collections import Counter
from pathlib import Path

from spindle import hub, reasoning as notes
from spindle.prompt import render_example
from . import app_actions, classification, combined, dates, extraction, tool_calling

SOURCES = {**tool_calling.SOURCES, **dates.SOURCES, **app_actions.SOURCES, **extraction.SOURCES, **classification.SOURCES}

# Share of the mix per task, and the preferred share of each source inside a task. When a source
# is missing or too small, the others fill its place in proportion.
TASKS = {'tools': 0.45, 'no_call': 0.15, 'extract': 0.25, 'classify': 0.15}
SHARES = {
    'tools': {'xlam': 0.38, 'toolace': 0.15, 'glaive': 0.15, 'app_actions': 0.14, 'dates': 0.08, 'clinc': 0.07,
              'banking77': 0.03},
    'no_call': {'when2call': 0.35, 'glaive': 0.22, 'clinc': 0.18, 'app_actions': 0.125, 'toolace': 0.085,
                'banking77': 0.04},
    'extract': {'massive': 0.25, 'few_nerd': 0.25, 'records': 0.25, 'pile_ner': 0.17, 'atis': 0.08},
    'classify': {'sst2': 0.25, 'tweet_sentiment': 0.25, 'banking77': 0.15, 'ag_news': 0.13, 'emotion': 0.12,
                 'clinc_labels': 0.10},
}


def usable(row, max_chars):
    """Cheap filters: sane shapes and a prompt that will likely fit the training length."""
    if not row['query'] or not all(isinstance(a.get('arguments'), dict) for a in row['answers']):
        return False
    prompt, target = render_example(row)
    return len(prompt) + len(target) <= max_chars and len(target) <= 700


def allocate(quota, shares, available):
    """Rows to take per source: preferred shares, with shortfalls spread over the other sources."""
    take = {name: 0 for name in shares}
    open_names, left = [n for n in shares if available.get(n)], quota
    while left > 0 and open_names:
        total = sum(shares[n] for n in open_names)
        spent = 0
        for name in list(open_names):
            amount = min(available[name] - take[name], max(1, round(left * shares[name] / total)))
            take[name] += amount
            spent += amount
            if take[name] >= available[name]:
                open_names.remove(name)
        left -= spent
        if not spent:
            break
    overshoot = sum(take.values()) - quota
    for name in sorted(take, key=take.get, reverse=True):
        cut = min(max(overshoot, 0), take[name])
        take[name], overshoot = take[name] - cut, overshoot - cut
    return take


def build(output, rows=128000, validation=256, max_chars=2000, seed=0, only=None, reasoning=False,
          combined_share=0.0):
    """`reasoning` adds a reasoning note to tool-calling and no-call rows (see spindle/reasoning.py).
    `combined_share` is the share of tool-calling rows that join two requests into one (combined.py)."""
    output = Path(output)
    rng = random.Random(seed)
    pools, dropped = {}, Counter()
    names = [n for n in SOURCES if not only or n in only]
    for number, name in enumerate(names, 1):
        print(f'[{number}/{len(names)}] {name}', flush=True)
        seen = set()
        for row in SOURCES[name]():
            key = (row['query'], row.get('system', ''), tuple(t['name'] for t in row['tools']))
            if key in seen or not usable(row, max_chars):
                dropped[name] += 1
                continue
            seen.add(key)
            pools.setdefault(row['task'], {}).setdefault(name, []).append(row)
        print('  ' + ', '.join(f'{task}: {len(pool[name]):,}' for task, pool in pools.items() if name in pool)
              + f'  (dropped {dropped[name]:,})', flush=True)

    shares = {task: dict(parts) for task, parts in SHARES.items()}
    if combined_share:
        made = [row for row in combined.generate(pools.get('tools', {}), round(rows * TASKS['tools'] * combined_share * 1.2),
                                                 seed) if usable(row, max_chars)]
        pools.setdefault('tools', {})['combined'] = made
        shares['tools'] = {**{k: v * (1 - combined_share) for k, v in shares['tools'].items()}, 'combined': combined_share}
        print(f'combined: {len(made):,} rows that join two requests', flush=True)

    train, held, table, xlam_calls = [], [], {}, 0
    for task, share in TASKS.items():
        pool = pools.get(task, {})
        quota = round(rows * share) + validation
        available = {n: len(r) for n, r in pool.items()}
        if task == 'no_call':
            # When2Call only has declines, on tools in the style of xLAM. Without enough xLAM calls
            # next to them the model learns to decline every tool of that style.
            available['when2call'] = min(available.get('when2call', 0), xlam_calls // 2)
        take = allocate(quota, shares[task], available)
        if task == 'tools':
            xlam_calls = take.get('xlam', 0)
        chosen = []
        for name, amount in take.items():
            if amount:
                chosen += rng.sample(pool[name], amount)
        for row in chosen:  # notes only where there is a tool decision to explain
            if reasoning and task in ('tools', 'no_call'):
                row.setdefault('reasoning', notes.note(row))
            else:
                row.pop('reasoning', None)
        rng.shuffle(chosen)
        held += chosen[:validation]
        chosen = chosen[validation:]
        # A position in [0, 1) per row: sorting by it spreads every task evenly over the file.
        train += [((i + rng.random()) / max(len(chosen), 1), row) for i, row in enumerate(chosen)]
        table[task] = {'rows': len(chosen), 'wanted': round(rows * share), 'sources': {k: v for k, v in take.items() if v}}
    train = [row for _, row in sorted(train, key=lambda item: item[0])]

    hub.write_jsonl(output / 'train.jsonl', train)
    hub.write_jsonl(output / 'validation.jsonl', held)
    summary = {'rows': len(train), 'validation_rows': len(held), 'seed': seed, 'max_chars': max_chars,
               'reasoning': reasoning, 'combined_share': combined_share,
               'tasks': table, 'dropped': dict(dropped)}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    print(f'\n{"task":<10} {"rows":>8} {"wanted":>8}  sources')
    for task, info in table.items():
        print(f'{task:<10} {info["rows"]:>8,} {info["wanted"]:>8,}  '
              + ', '.join(f'{k} {v:,}' for k, v in info['sources'].items()))
    print(f'Wrote {len(train):,} training rows and {len(held):,} validation rows to {output}', flush=True)
    return summary
