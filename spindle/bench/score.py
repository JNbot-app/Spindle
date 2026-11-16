"""Scoring. Every case gives a small record; `summarize` turns the records of one set into a score."""


def normal(value):
    """Values compared loosely: case and spacing of text, and 5 == 5.0."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return ' '.join(value.lower().split())
    if isinstance(value, (list, tuple)):
        return [normal(v) for v in value]
    if isinstance(value, dict):
        return {str(k): normal(v) for k, v in sorted(value.items())}
    return value


def _fields(calls):
    return {(c['name'], key, repr(normal(value))) for c in calls for key, value in c['arguments'].items()}


def _ast_call(call, truth):
    (name, accepted), = truth.items()
    if call['name'] != name or any(key not in accepted for key in call['arguments']):
        return False
    for key, options in accepted.items():
        options = [normal(o) for o in options]
        if key not in call['arguments']:
            if '' not in options:
                return False
        elif normal(call['arguments'][key]) not in options:
            return False
    return True


def _ast(calls, expected):
    """BFCL rule: every expected call is matched by a different output call, in any order."""
    if len(calls) != len(expected):
        return False
    free = list(calls)
    for truth in expected:
        match = next((c for c in free if _ast_call(c, truth)), None)
        if match is None:
            return False
        free.remove(match)
    return True


def score(case, calls):
    kind, expected = case['kind'], case['expected']
    record = {'id': case['id'], 'group': case.get('group', ''), 'called': bool(calls),
              'should_call': expected if kind == 'decision' else bool(expected)}
    if kind == 'calls':
        record['correct'] = normal(calls) == normal(expected)
    elif kind == 'ast':
        record['correct'] = _ast(calls, expected)
    elif kind == 'decision':
        record['correct'] = bool(calls) == expected
    elif kind == 'fields':
        got, want = _fields(calls), _fields(expected)
        record.update(tp=len(got & want), fp=len(got - want), fn=len(want - got), correct=got == want)
    else:
        raise ValueError(f'Unknown case kind {kind}')
    return record


def summarize(records):
    """{score, metric, cases, call_rate_when_needed, decline_rate_when_not, groups} for one set."""
    def one(rows):
        if any('tp' in r for r in rows):
            tp, fp, fn = (sum(r[k] for r in rows) for k in ('tp', 'fp', 'fn'))
            return 200 * tp / max(2 * tp + fp + fn, 1), 'field F1'
        return 100 * sum(r['correct'] for r in rows) / max(len(rows), 1), 'accuracy'

    value, metric = one(records)
    need = [r for r in records if r['should_call']]
    rest = [r for r in records if not r['should_call']]
    groups = {}
    for record in records:
        if record['group']:
            groups.setdefault(record['group'], []).append(record)
    return {'score': round(value, 1), 'metric': metric, 'cases': len(records),
            'call_rate_when_needed': round(100 * sum(r['called'] for r in need) / len(need), 1) if need else None,
            'decline_rate_when_not': round(100 * sum(not r['called'] for r in rest) / len(rest), 1) if rest else None,
            'groups': {name: {'score': round(one(rows)[0], 1), 'cases': len(rows)}
                       for name, rows in sorted(groups.items())}}
