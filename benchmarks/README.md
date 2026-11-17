# Benchmark results

`results/` holds one summary per model: the output of `spindle bench --output` without the per-case part,
plus `name`, `size_mb` and `order` for the charts. `plot.py` turns them into `docs/images/benchmarks.svg`,
`docs/images/size.svg` and prints the Markdown table used in the README.

```sh
spindle bench models/spindle.cact --quick --output report.json
python - <<'EOF'
import json
report = json.load(open('report.json'))
report.pop('cases')
report.update(name='My model', size_mb=75.0, order=4, chart=False)
json.dump(report, open('benchmarks/results/my-model.json', 'w'), indent=1)
EOF
python benchmarks/plot.py
```

How each set is scored and how each model was run: [docs/benchmarks.md](../docs/benchmarks.md).
