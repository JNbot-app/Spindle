"""Tool-list variety: the same request with a different tool list.

In most tool-calling datasets the tool list is almost fixed for each kind of request, so the model
can answer from the request alone and never learns to read the list. It then calls a tool that is
not there, and extra tools in the prompt cost accuracy. Two edits make the list a real input. Both
keep the request text:

* varied:  a call row keeps its needed tools, loses some unused ones (each stays with chance 0.5),
           gets 1-3 tools from other rows and a new order. The answer does not change.
* absent:  a copy of a single-call row without its needed tool and with a varied list; the answer
           becomes [] (no call). The original row stays.

Every added tool comes from other rows, where it is a right answer, so "an unfamiliar tool" or "an
unusual list" says nothing about the answer. Varied lists alone change little; the absent rows teach
the model to check that the needed tool is listed. Tools that could do the same job as a needed tool
(same name word, or a synonym group below) are never added next to it or kept in its place.
"""
import copy
import json
import random

from spindle.reasoning import decline_note

MAX_ADDED = 3
KEEP_UNUSED = .5
# Common verbs: sharing one of these does not make two tools interchangeable.
VERBS = {'get', 'set', 'send', 'create', 'update', 'delete', 'add', 'remove', 'list', 'check', 'make',
         'start', 'stop', 'run', 'record', 'log', 'find', 'search', 'show', 'open', 'close', 'turn', 'do'}
# Words of tools that could serve the same request.
SYNONYMS = (
    {'chat', 'message', 'msg', 'email', 'mail', 'sms', 'text', 'whisper', 'say', 'announce', 'broadcast', 'notify'},
    {'pay', 'payment', 'buy', 'sell', 'sale', 'trade', 'order', 'purchase', 'shop', 'transfer'},
    {'timer', 'alarm', 'reminder', 'cooldown', 'countdown'},
    {'navigate', 'move', 'go', 'goto', 'travel', 'route', 'directions', 'follow', 'teleport', 'waypoint'},
    {'weather', 'forecast'},
    {'event', 'calendar', 'meeting', 'appointment', 'schedule'},
    {'status', 'health', 'telemetry', 'metric', 'metrics'},
    {'classify', 'intent', 'sentiment', 'label'},
)


def _key(tool):
    return json.dumps(tool, sort_keys=True)


def _words(name):
    return set(name.lower().replace('-', '_').split('_')) - VERBS


def _conflicts(name, needed):
    """True when tool `name` is a needed tool or could do the same job as one."""
    words = _words(name)
    for other in needed:
        if name == other:
            return True
        other_words = _words(other)
        if words & other_words or any(words & group and other_words & group for group in SYNONYMS):
            return True
    return False


def tool_pool(rows):
    """Every distinct tool definition of the data, by name."""
    pool = {}
    for row in rows:
        for tool in row.get('tools', []):
            pool.setdefault(tool['name'], {})[_key(tool)] = tool
    return {name: list(variants.values()) for name, variants in sorted(pool.items())}


def _new_list(tools, needed, pool, rng, added):
    kept = [tool for tool in tools
            if tool['name'] in needed or (not _conflicts(tool['name'], needed) and rng.random() < KEEP_UNUSED)]
    taken = {tool['name'] for tool in kept}
    free = [name for name in pool if name not in taken and not _conflicts(name, needed)]
    kept += [copy.deepcopy(rng.choice(pool[name])) for name in rng.sample(free, min(added, len(free)))]
    rng.shuffle(kept)
    return kept


def _edited(row, tools, needed, answers, pool, rng, fits):
    """A copy of `row` with a new list built from `tools`; fewer added tools on each retry. None if none fits."""
    for added in range(rng.randint(1, MAX_ADDED), -1, -1):
        candidate = {**row, 'answers': answers, 'tools': _new_list(tools, needed, pool, rng, added)}
        if candidate['tools'] and candidate['tools'] != row.get('tools') and fits(candidate):
            return candidate
    return None


def varied(row, pool, rng, fits=lambda row: True):
    """A call row with a different tool list and the same answer. None if no list fits."""
    needed = {call['name'] for call in row['answers']}
    if not needed:
        return None
    return _edited(row, row.get('tools', []), needed, row['answers'], pool, rng, fits)


def absent(row, pool, rng, fits=lambda row: True):
    """A single-call row without its needed tool, so the answer is []. None if unusable."""
    if len(row['answers']) != 1:
        return None
    needed = {row['answers'][0]['name']}
    tools = [tool for tool in row.get('tools', []) if tool['name'] not in needed]
    if row.get('reasoning'):  # the note described the call that is now impossible
        row = {**row, 'reasoning': decline_note(row['query'])}
    return _edited(row, tools, needed, [], pool, rng, fits)


def augment(rows, variety, absent_share, seed=0, fits=lambda row: True):
    """Rows with varied tool lists (each call row with chance `variety`) plus tool-absent rows
    (`absent_share` x the row count). `fits(row)` rejects edits, e.g. prompts that get too long."""
    rng = random.Random(seed)
    pool = tool_pool(rows)
    singles = [row for row in rows if len(row['answers']) == 1]
    rng.shuffle(singles)
    wanted, made = round(absent_share * len(rows)), []
    for row in singles:
        if len(made) == wanted:
            break
        edited = absent(row, pool, rng, fits)
        if edited is not None:
            made.append(edited)
    if len(made) < wanted:
        print(f'  tool lists: only {len(made):,} of {wanted:,} tool-absent rows could be built', flush=True)
    out, changed = [], 0
    for index, row in enumerate(rows):
        if row['answers'] and rng.random() < variety:
            edited = varied(row, pool, rng, fits)
            if edited is not None:
                row, changed = edited, changed + 1
        out.append(row)
        if (index + 1) % 10000 == 0:
            print(f'  tool lists: {index + 1:,}/{len(rows):,} rows, {changed:,} varied', flush=True)
    print(f'  tool lists: {changed:,} rows with a varied list, {len(made):,} tool-absent rows added '
          f'({len(pool):,} distinct tools)', flush=True)
    return out + made
