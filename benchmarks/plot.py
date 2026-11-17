"""Draw the benchmark charts in docs/images/ from the result files in benchmarks/results/.

    python benchmarks/plot.py

Each result file is a `spindle bench --output` report (the per-case part may be left out) with three
extra keys: `name` (shown in the legend), `size_mb` (model file size) and `order` (legend position).
`"chart": false` keeps a model in the table but out of the charts.
The charts are plain SVG with no dependencies. They follow the reader's light or dark theme.
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'benchmarks' / 'results'
IMAGES = ROOT / 'docs' / 'images'

# Sets in the order shown, grouped by task, with the short metric name.
TASKS = [
    ('Tool calling', [('mobile_actions', 'Mobile Actions', 'exact'), ('droidcall', 'DroidCall', 'exact'),
                      ('bfcl', 'BFCL v4', 'AST')]),
    ('Knowing when not to call', [('when2call', 'When2Call', 'acc.'), ('tool_removed', 'Tool removed', 'acc.')]),
    ('Extraction', [('dstc8', 'DSTC8', 'F1'), ('snips_gold', 'SNIPS', 'F1'), ('snips_7way', 'SNIPS 7-way', 'F1')]),
    ('Classification', [('rotten_tomatoes', 'Rotten Tomatoes', 'acc.'), ('dbpedia', 'DBpedia', 'acc.')]),
]
SETS = [key for _, rows in TASKS for key, _, _ in rows]

# Categorical slots 1-3 (blue, orange, aqua), light and dark steps; checked for colour-blind separation.
SERIES = [('#2a78d6', '#3987e5'), ('#eb6834', '#d95926'), ('#1baf7a', '#199e70')]
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"


def load():
    models = [json.loads(path.read_text()) for path in sorted(RESULTS.glob('*.json'))]
    models.sort(key=lambda model: model.get('order', 99))
    charted = [model for model in models if model.get('chart', True)]
    if len(charted) > len(SERIES):
        raise SystemExit(f'At most {len(SERIES)} models per chart (found {len(charted)})')
    return models, charted


def mean_score(model):
    scores = [model['results'][key]['score'] for key in SETS if key in model['results']]
    return sum(scores) / len(scores) if len(scores) == len(SETS) else None


def seconds(model):
    """Average time per request over all sets."""
    values = [model['results'][key]['seconds_per_case'] for key in SETS if key in model['results']]
    return sum(values) / len(values) if values else None


def style(count):
    light = ''.join(f'--s{i}:{light};' for i, (light, _) in enumerate(SERIES[:count]))
    dark = ''.join(f'--s{i}:{dark};' for i, (_, dark) in enumerate(SERIES[:count]))
    fills = ''.join(f'.s{i}{{fill:var(--s{i})}}' for i in range(count))
    return (f'<style>svg{{--ink:#1f2328;--ink2:#59636e;--muted:#818b98;--grid:#d1d9e0;{light}}}'
            f'@media (prefers-color-scheme: dark){{svg{{--ink:#f0f6fc;--ink2:#9198a1;--muted:#757d86;'
            f'--grid:#3d444d;{dark}}}}}'
            f'text{{font-family:{FONT};fill:var(--ink)}}.ink2{{fill:var(--ink2)}}.muted{{fill:var(--muted)}}'
            f'.grid{{stroke:var(--grid);stroke-width:1}}{fills}</style>')


def legend(models, x, y):
    parts, cursor = [], x
    for i, model in enumerate(models):
        parts.append(f'<rect class="s{i}" x="{cursor}" y="{y - 9}" width="10" height="10" rx="2"/>'
                     f'<text x="{cursor + 15}" y="{y}" font-size="12.5">{model["name"]}</text>')
        cursor += 15 + 7.2 * len(model['name']) + 22
    return ''.join(parts)


def bar(x, y, width, height, cls):
    """A bar with a rounded data end and a square base."""
    if width <= 0:
        return ''
    r = min(3, width, height / 2)
    return (f'<path class="{cls}" d="M{x},{y} H{x + width - r} Q{x + width},{y} {x + width},{y + r} '
            f'V{y + height - r} Q{x + width},{y + height} {x + width - r},{y + height} H{x} Z"/>')


def scores_chart(models):
    left, plot, right = 178, 520, 44
    bar_h, bar_gap, group_gap, section_gap = 8, 2, 13, 20
    width = left + plot + right
    y = 58
    body = []
    for title, rows in TASKS:
        y += section_gap
        body.append(f'<text x="0" y="{y}" font-size="12" font-weight="600" class="ink2">{title.upper()}</text>')
        y += 12
        for key, label, metric in rows:
            group = len(models) * bar_h + (len(models) - 1) * bar_gap
            middle = y + group / 2 + 4
            body.append(f'<text x="0" y="{middle}" font-size="13">{label}</text>'
                        f'<text x="{left - 14}" y="{middle}" font-size="11" text-anchor="end" class="muted">'
                        f'{metric}</text>')
            for i, model in enumerate(models):
                top = y + i * (bar_h + bar_gap)
                result = model['results'].get(key)
                if result is None:
                    body.append(f'<text x="{left + 4}" y="{top + bar_h - 0.5}" font-size="10" class="muted">'
                                f'not run</text>')
                    continue
                length = plot * result['score'] / 100
                body.append(bar(left, top, length, bar_h, f's{i}'))
                body.append(f'<text x="{left + length + 5}" y="{top + bar_h - 0.5}" font-size="10" '
                            f'class="ink2">{result["score"]:.0f}</text>')
            y += group + group_gap
    height = y + 22
    grid = ''.join(f'<line class="grid" x1="{left + plot * v / 100}" x2="{left + plot * v / 100}" y1="66" '
                   f'y2="{y - 6}"/><text x="{left + plot * v / 100}" y="{y + 10}" font-size="10.5" '
                   f'text-anchor="middle" class="muted">{v}</text>' for v in (0, 25, 50, 75, 100))
    cases = max(r['cases'] for m in models for r in m['results'].values())
    head = (f'<text x="0" y="18" font-size="16" font-weight="600">Score per benchmark</text>'
            f'<text x="0" y="37" font-size="12" class="ink2">Same cases and scoring for every model, '
            f'{cases} cases per set, higher is better. None of these sets is used in training.</text>'
            + legend(models, left, 58))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
            f'height="{height}" role="img" aria-label="Benchmark scores per model">'
            f'{style(len(models))}{head}{grid}{"".join(body)}</svg>\n')


def size_chart(models):
    """Mean score against file size (log scale)."""
    left, top, plot_w, plot_h = 52, 76, 560, 230
    width, height = left + plot_w + 150, top + plot_h + 48
    sizes = [m['size_mb'] for m in models]
    low, high = 10 ** math.floor(math.log10(min(sizes) * 0.7)), 10 ** math.ceil(math.log10(max(sizes) * 1.4))
    y_low, y_high = 0, 70

    def px(size):
        return left + plot_w * (math.log10(size) - math.log10(low)) / (math.log10(high) - math.log10(low))

    def py(score):
        return top + plot_h * (1 - (score - y_low) / (y_high - y_low))

    parts = [f'<text x="0" y="18" font-size="16" font-weight="600">Accuracy against size</text>',
             f'<text x="0" y="37" font-size="12" class="ink2">Mean score over all ten sets, against the size of '
             f'the model file (log scale). Time is the average per request on an Apple M1.</text>']
    for v in range(y_low, y_high + 1, 10):
        parts.append(f'<line class="grid" x1="{left}" x2="{left + plot_w}" y1="{py(v)}" y2="{py(v)}"/>'
                     f'<text x="{left - 8}" y="{py(v) + 3.5}" font-size="10.5" text-anchor="end" '
                     f'class="muted">{v}</text>')
    tick = low
    while tick <= high:
        label = f'{tick / 1000:g} GB' if tick >= 1000 else f'{tick:g} MB'
        parts.append(f'<text x="{px(tick)}" y="{top + plot_h + 18}" font-size="10.5" text-anchor="middle" '
                     f'class="muted">{label}</text>')
        tick *= 10
    parts.append(f'<text x="{left + plot_w / 2}" y="{top + plot_h + 38}" font-size="11.5" text-anchor="middle" '
                 f'class="ink2">Model file size</text>'
                 f'<text x="{left - 36}" y="{top - 12}" font-size="11.5" class="ink2">Mean score</text>')
    points = [(i, model, px(model['size_mb']), py(mean_score(model)))
              for i, model in enumerate(models) if mean_score(model) is not None]
    for i, model, x, y in points:
        size = f'{model["size_mb"] / 1000:.1f} GB' if model['size_mb'] >= 1000 else f'{model["size_mb"]:.0f} MB'
        # Label to the right of the dot, or below it when another dot sits close on the right.
        crowded = any(0 < ox - x < 150 and abs(oy - y) < 40 for _, _, ox, oy in points)
        lx, ly = (x - 6, y + 24) if crowded else (x + 12, y - 2)
        parts.append(f'<circle class="s{i}" cx="{x}" cy="{y}" r="6.5"/>'
                     f'<text x="{lx}" y="{ly}" font-size="13" font-weight="600">{model["name"]}</text>'
                     f'<text x="{lx}" y="{ly + 15}" font-size="11" class="ink2">{mean_score(model):.1f} mean · '
                     f'{size} · {seconds(model):.1f} s per request</text>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
            f'height="{height}" role="img" aria-label="Mean score against model size">'
            f'{style(len(models))}{"".join(parts)}</svg>\n')


def table(models):
    """The Markdown table for the README, so the numbers are never typed by hand."""
    head = '| Benchmark | Metric | ' + ' | '.join(m['name'] for m in models) + ' |'
    lines = [head, '| --- | --- |' + ' ---: |' * len(models)]
    for _, rows in TASKS:
        for key, label, metric in rows:
            cells = []
            best = max((m['results'][key]['score'] for m in models if key in m['results']), default=None)
            for m in models:
                result = m['results'].get(key)
                text = '–' if result is None else f'{result["score"]:.1f}'
                cells.append(f'**{text}**' if result and result['score'] == best else text)
            lines.append(f'| {label} | {metric} | ' + ' | '.join(cells) + ' |')
    means = [mean_score(m) for m in models]
    top = max((v for v in means if v is not None), default=None)
    lines.append('| **Mean** | | ' + ' | '.join('–' if v is None else f'**{v:.1f}**' if v == top else f'{v:.1f}'
                                             for v in means) + ' |')
    lines.append('| Time per request (M1) | | ' + ' | '.join(f'{seconds(m):.2f} s' for m in models) + ' |')
    lines.append('| File size | | ' + ' | '.join(f'{m["size_mb"]:.0f} MB' if m['size_mb'] < 1000
                                              else f'{m["size_mb"] / 1000:.1f} GB' for m in models) + ' |')
    return '\n'.join(lines)


def main():
    models, charted = load()
    IMAGES.mkdir(parents=True, exist_ok=True)
    (IMAGES / 'benchmarks.svg').write_text(scores_chart(charted))
    (IMAGES / 'size.svg').write_text(size_chart(charted))
    print(table(models))
    print(f'\nWrote {IMAGES / "benchmarks.svg"} and {IMAGES / "size.svg"}')


if __name__ == '__main__':
    main()
