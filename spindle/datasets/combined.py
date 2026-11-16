"""Rows that join two single-call requests into one request with two calls.

Public sets with several calls mostly repeat one tool. Real requests often ask for two different
things ("turn on the flashlight and email Sam ..."), and a model that has not seen that stops after
the first call. These rows are built from rows of the other sources, so nothing new is invented.
"""
import random

JOINS = ('{a} Also, {b}', '{a} And {b}', '{a} Then {b}', '{a} After that, {b}', '{a} {b}',
         'First, {a} Second, {b}', '{a} One more thing: {b}')


def _lower_first(text):
    return text[:1].lower() + text[1:] if text[:2].istitle() or text[:1].isupper() and text[1:2] == ' ' else text


def _join(first, second, rng):
    """One row from two single-call rows, or None when their tools clash."""
    names = {tool['name'] for tool in first['tools']}
    if first['answers'][0]['name'] == second['answers'][0]['name'] or (first.get('system') and second.get('system')):
        return None
    if any(tool['name'] in names for tool in second['tools']):
        return None
    a = first['query'].strip()
    a = a if a[-1:] in '.!?' else a + '.'
    pattern = rng.choice(JOINS)
    b = second['query'].strip()
    tools = first['tools'] + second['tools']
    rng.shuffle(tools)
    row = {'query': pattern.format(a=_lower_first(a) if pattern.startswith('First') else a,
                                   b=_lower_first(b) if ', {b}' in pattern else b),
           'tools': tools, 'answers': first['answers'] + second['answers'], 'task': 'tools', 'source': 'combined'}
    system = first.get('system') or second.get('system')
    if system:
        row['system'] = system
    notes = [r.get('reasoning') for r in (first, second)]
    if all(notes):
        row['reasoning'] = ' '.join(notes)
    return row


def generate(pools, count, seed=0):
    """`count` joined rows from the single-call rows of `pools` ({source: rows})."""
    rng = random.Random(seed)
    singles = [row for rows in pools.values() for row in rows
               if len(row['answers']) == 1 and len(row['tools']) <= 4 and len(row['query']) <= 300]
    made, tries = [], 0
    while len(made) < count and tries < count * 5 and len(singles) > 1:
        tries += 1
        row = _join(*rng.sample(singles, 2), rng)
        if row:
            made.append(row)
    return made
