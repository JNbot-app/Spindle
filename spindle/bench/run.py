"""Run benchmark sets through a model and print one table.

Two backends answer a case with a list of calls:
  * SpindleBackend: a .cact model through the native engine,
  * OpenAIBackend: any server with the OpenAI chat API and tool calling (llama.cpp's
    `llama-server`, vLLM, Ollama, ...), so other small models are scored by the same code.
"""
import json
import time
import urllib.request
from pathlib import Path

from . import sets
from .score import score, summarize

QUICK = 200  # cases per set with --quick


class SpindleBackend:
    def __init__(self, model, *, window=0, filters=True, reasoning=True, library=None, threads=0,
                 max_new_tokens=320):
        from spindle import Spindle
        self.model = Spindle(model, library=library, threads=threads, filters=filters, window=window,
                             reasoning=reasoning)
        self.max_new_tokens = max_new_tokens
        self.current = None

    def __call__(self, case):
        key = (json.dumps(case['tools'], sort_keys=True), case.get('system', ''))
        if key != self.current:
            self.model.set_tools(case['tools'], system=case.get('system', ''))
            self.current = key
        result = self.model(case['query'], max_new_tokens=self.max_new_tokens,
                            require_call=case.get('require_call', False))
        return result['function_calls']

    def close(self):
        self.model.close()


class OpenAIBackend:
    def __init__(self, url, model='', *, key='', max_new_tokens=512):
        self.url, self.model, self.key, self.max_new_tokens = url.rstrip('/'), model, key, max_new_tokens

    def __call__(self, case):
        messages = [{'role': 'system', 'content': case['system']}] if case.get('system') else []
        messages.append({'role': 'user', 'content': case['query']})
        body = {'model': self.model, 'messages': messages, 'temperature': 0, 'max_tokens': self.max_new_tokens,
                'tools': [{'type': 'function', 'function': tool} for tool in case['tools']]}
        if case.get('require_call'):
            body['tool_choice'] = 'required'
        headers = {'Content-Type': 'application/json'}
        if self.key:
            headers['Authorization'] = f'Bearer {self.key}'
        request = urllib.request.Request(f'{self.url}/chat/completions', json.dumps(body).encode(), headers)
        with urllib.request.urlopen(request, timeout=600) as response:
            message = json.load(response)['choices'][0]['message']
        calls = []
        for item in message.get('tool_calls') or []:
            arguments = item['function'].get('arguments') or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            calls.append({'name': item['function']['name'], 'arguments': arguments})
        return calls

    def close(self):
        pass


def run_set(backend, name, limit=0):
    cases = sets.load(name, limit)
    records, started, errors = [], time.monotonic(), 0
    for index, case in enumerate(cases, 1):
        try:
            calls, error = backend(case), ''
        except Exception as problem:  # one bad case (prompt too long, bad schema) must not stop the run
            calls, error, errors = [], f'{type(problem).__name__}: {problem}', errors + 1
        record = score(case, calls)
        record.update(output=calls, error=error)
        records.append(record)
        if index % 25 == 0 or index == len(cases):
            elapsed = time.monotonic() - started
            print(f'  {name}: {index}/{len(cases)}  score so far {summarize(records)["score"]}  '
                  f'{elapsed / index:.2f} s/case  left {elapsed / index * (len(cases) - index) / 60:.1f} min',
                  flush=True)
    summary = summarize(records)
    summary.update(errors=errors, seconds_per_case=round((time.monotonic() - started) / max(len(cases), 1), 3))
    return summary, records


def table(results):
    lines = [f'{"set":<16} {"cases":>5} {"score":>6}  {"metric":<9} {"calls when needed":>17} {"declines when not":>17}']
    for name, summary in results.items():
        def rate(value):
            return '-' if value is None else f'{value:.1f}%'
        lines.append(f'{name:<16} {summary["cases"]:>5} {summary["score"]:>6.1f}  {summary["metric"]:<9} '
                     f'{rate(summary["call_rate_when_needed"]):>17} {rate(summary["decline_rate_when_not"]):>17}')
    return '\n'.join(lines)


def run(backend, names, *, limit=0, output=None, label=''):
    """Score `backend` on each named set; returns {set: summary} and writes a JSON report."""
    results, details = {}, {}
    for number, name in enumerate(names, 1):
        print(f'[{number}/{len(names)}] {name}', flush=True)
        results[name], details[name] = run_set(backend, name, limit)
        if output:  # after every set, so a failure later in the run keeps what is done
            output = Path(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({'model': label, 'limit': limit, 'results': results, 'cases': details},
                                         ensure_ascii=False, indent=1) + '\n')
    print('\n' + table(results), flush=True)
    if output:
        print(f'Report: {output}', flush=True)
    return results
