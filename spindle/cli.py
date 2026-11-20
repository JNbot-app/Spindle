"""Command line: `spindle run`, `spindle finetune`, `spindle bench`, `spindle pick-skip`, `spindle monarch`."""
import argparse
import json
from dataclasses import fields


def _run(args):
    from spindle import Spindle
    with open(args.tools, encoding='utf-8') as handle:
        tools = json.load(handle)
    with Spindle(args.model, tools, system=args.system, library=args.library, threads=args.threads,
                 filters=not args.no_filters, min_call_probability=args.min_call_probability,
                 window=args.window, decode_skip=args.decode_skip) as model:
        queries = args.query or []
        if not queries:
            print('Type a request (Ctrl-D to quit).')
        for query in queries or _stdin():
            result = model(query, max_new_tokens=args.max_new_tokens)
            if not args.metrics:
                result.pop('metrics')
            print(json.dumps(result, indent=2, ensure_ascii=False))


def _stdin():
    while True:
        try:
            line = input('> ').strip()
        except EOFError:
            return
        if line:
            yield line


def _finetune(args):
    from spindle.training.config import TrainConfig
    from spindle.training.finetune import finetune
    config = TrainConfig(**{f.name: getattr(args, f.name) for f in fields(TrainConfig)})
    finetune(args.model, args.train, args.output, validation_path=args.validation, config=config)


def _bench(args):
    from spindle.bench import run, sets
    names = args.sets or list(sets.SETS)
    if args.server:
        backend = run.OpenAIBackend(args.server, args.model, key=args.key)
    else:
        backend = run.SpindleBackend(args.model, window=args.window, filters=not args.no_filters,
                                     reasoning=not args.no_reasoning, library=args.library, threads=args.threads,
                                     decode_skip=args.decode_skip, debias_enums=args.debias_enums)
    try:
        run.run(backend, names, limit=args.limit or (run.QUICK if args.quick else 0), output=args.output,
                label=args.model)
    finally:
        backend.close()


def _build_data(args):
    from spindle.datasets import mix
    mix.build(args.output, rows=args.rows, validation=args.validation, max_chars=args.max_chars, seed=args.seed,
              only=args.sources, reasoning=args.reasoning, combined_share=args.combined)


def _pick_skip(args):
    from spindle.inference import decode_skip
    decode_skip.main(args.model, count=args.layers, data=args.data, per_set=args.per_set, pool=args.pool,
                     window=args.window, threads=args.threads, library=args.library)


def _monarch(args):
    from spindle.archive import to_monarch
    print(f'Wrote {to_monarch(args.source, args.output)}')


DECODE_SKIP_HELP = ('Faster decoding: generated tokens skip some layers (off by default). Alone it uses the '
                    "layers the model was trained to skip; or name them, e.g. 5-8,10")


def main(argv=None):
    parser = argparse.ArgumentParser(prog='spindle', description='Small on-device tool-calling model.')
    commands = parser.add_subparsers(dest='command', required=True)

    run = commands.add_parser('run', help='Call tools for one or more requests')
    run.add_argument('model', help='.cact model file')
    run.add_argument('--tools', required=True, help='JSON file with a list of tool definitions')
    run.add_argument('query', nargs='*', help='Requests (reads stdin when none are given)')
    run.add_argument('--system', default='', help='Optional system prompt')
    run.add_argument('--max-new-tokens', type=int, default=256)
    run.add_argument('--min-call-probability', type=float, default=0.0,
                     help='Suppress calls whose mean token probability is below this')
    run.add_argument('--no-filters', action='store_true',
                     help='Return every decoded call, also ones the request does not support')
    run.add_argument('--window', type=int, default=0,
                     help="Attention window in tokens (0: the model's own)")
    run.add_argument('--decode-skip', nargs='?', const=True, metavar='LAYERS', help=DECODE_SKIP_HELP)
    run.add_argument('--threads', type=int, default=0, help='Worker threads (0: automatic)')
    run.add_argument('--library', help='Path to the native engine library (default: build/native)')
    run.add_argument('--metrics', action='store_true', help='Also print token counts and timing')
    run.set_defaults(handler=_run)

    from spindle.training.config import TrainConfig
    tune = commands.add_parser('finetune', help='Fine-tune a model on JSONL tool-calling data')
    tune.add_argument('--model', required=True, help='Parent .cact model')
    tune.add_argument('--train', required=True, help='Training JSONL (query, tools, answers)')
    tune.add_argument('--validation', help='Validation JSONL (default: hold out rows from --train)')
    tune.add_argument('--output', required=True, help='Output folder (model.cact, training-log.jsonl, run.json)')
    for f in fields(TrainConfig):
        kind = (lambda v: v.lower() in ('1', 'true', 'yes')) if isinstance(f.default, bool) else type(f.default)
        tune.add_argument('--' + f.name.replace('_', '-'), type=kind, default=f.default)
    tune.set_defaults(handler=_finetune)

    from spindle.bench.sets import SETS
    bench = commands.add_parser('bench', help='Score a model on public benchmark sets')
    bench.add_argument('model', help='.cact model file, or the model name when --server is given')
    bench.add_argument('--sets', nargs='*', choices=list(SETS), help='Sets to run (default: all)')
    bench.add_argument('--quick', action='store_true', help='200 cases per set')
    bench.add_argument('--limit', type=int, default=0, help='Cases per set (0: all)')
    bench.add_argument('--output', help='Write a JSON report with every case')
    bench.add_argument('--server', help='Score an OpenAI-compatible server instead, e.g. http://127.0.0.1:8080/v1')
    bench.add_argument('--key', default='', help='API key for --server')
    bench.add_argument('--window', type=int, default=1024, help="Attention window (0: the model's own)")
    bench.add_argument('--no-filters', action='store_true', help='Score the raw decoded calls')
    bench.add_argument('--no-reasoning', action='store_true', help='Never let the model write a note before its calls')
    bench.add_argument('--decode-skip', nargs='?', const=True, metavar='LAYERS', help=DECODE_SKIP_HELP)
    bench.add_argument('--debias-enums', action='store_true',
                       help='Correct the pull towards some labels of enum lists, learned from earlier cases')
    bench.add_argument('--threads', type=int, default=0, help='Worker threads (0: automatic)')
    bench.add_argument('--library', help='Path to the native engine library')
    bench.set_defaults(handler=_bench)

    data = commands.add_parser('build-data', help='Build the general training mix from public datasets')
    data.add_argument('--output', required=True, help='Folder for train.jsonl, validation.jsonl, summary.json')
    data.add_argument('--rows', type=int, default=128000, help='Training rows')
    data.add_argument('--validation', type=int, default=256, help='Held-out rows per task')
    data.add_argument('--max-chars', type=int, default=2000, help='Longest prompt + answer kept, in characters')
    data.add_argument('--seed', type=int, default=0)
    data.add_argument('--sources', nargs='*', help='Only these sources (default: all)')
    data.add_argument('--reasoning', action='store_true',
                      help='Add a short reasoning note before the calls of tool-calling and no-call rows')
    data.add_argument('--combined', type=float, default=0.0,
                      help='Share of tool-calling rows that join two requests into one (several calls)')
    data.set_defaults(handler=_build_data)

    pick = commands.add_parser('pick-skip', help='Measure which layers generated tokens can skip (decode skip)')
    pick.add_argument('model', help='.cact model file')
    pick.add_argument('--layers', type=int, default=7, help='How many layers to pick')
    pick.add_argument('--data', help='JSONL with {query, tools, [system]} per line (default: benchmark requests)')
    pick.add_argument('--per-set', type=int, default=6, help='Requests per benchmark set without --data')
    pick.add_argument('--pool', type=int, default=6, help='Layers tried per step after the first')
    pick.add_argument('--window', type=int, default=0, help="Attention window (0: the model's own)")
    pick.add_argument('--threads', type=int, default=0, help='Worker threads (0: automatic)')
    pick.add_argument('--library', help='Path to the native engine library')
    pick.set_defaults(handler=_pick_skip)

    monarch = commands.add_parser('monarch', help='Convert Kronecker Hadamard factors to Monarch (same outputs)')
    monarch.add_argument('source')
    monarch.add_argument('output')
    monarch.set_defaults(handler=_monarch)

    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == '__main__':
    main()
