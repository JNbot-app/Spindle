"""Command line: `spindle run`, `spindle finetune`, `spindle monarch`."""
import argparse
import json
from dataclasses import fields


def _run(args):
    from spindle import Spindle
    with open(args.tools, encoding='utf-8') as handle:
        tools = json.load(handle)
    with Spindle(args.model, tools, system=args.system, library=args.library, threads=args.threads,
                 filters=not args.no_filters, min_call_probability=args.min_call_probability,
                 window=args.window) as model:
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


def _monarch(args):
    from spindle.archive import to_monarch
    print(f'Wrote {to_monarch(args.source, args.output)}')


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
                     help="Attention window in tokens (0: the model's own); 1024 helps with long tool lists")
    run.add_argument('--threads', type=int, default=4)
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
        tune.add_argument('--' + f.name.replace('_', '-'), type=type(f.default), default=f.default)
    tune.set_defaults(handler=_finetune)

    monarch = commands.add_parser('monarch', help='Convert Kronecker Hadamard factors to Monarch (same outputs)')
    monarch.add_argument('source')
    monarch.add_argument('output')
    monarch.set_defaults(handler=_monarch)

    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == '__main__':
    main()
