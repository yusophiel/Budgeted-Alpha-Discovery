# Budgeted Adaptive Falsification for Equity Alpha Discovery

A Python research framework for deciding **which robustness test to run next** when validating equity alpha candidates under a limited budget.

The pipeline generates symbolic factors, evaluates them in walk-forward windows, and compares adaptive test selection with fixed, random, and cost-based policies. It tracks survivor precision, recall, false eliminations, and research cost. An oracle policy provides a hindsight reference.

## Quick start

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m adaptive_alpha.cli build-benchmark
```

The default demo uses a fixed seed and small, generated dataset. No API key or external data is needed. Demo results illustrate the workflow; they are not evidence of investment performance.

Results are saved to `artifacts/demo/`:

- `report.md` — readable policy comparison.
- `policy_results.csv` — policy metrics.
- `benchmark.csv` — factor-by-test evaluations.
- Additional CSVs and a manifest record training splits and decisions.

Adjust experiment size, walk-forward windows, and budget in [`configs/demo.json`](configs/demo.json).

## How it works

1. Generate symbolic price and volume factors.
2. Evaluate candidates across search, falsification, and future windows.
3. Train test-selection models on earlier benchmark episodes.
4. Compare policies on held-out episodes under the same budget.

## Code map

| File | Responsibility |
| --- | --- |
| `adaptive_alpha/data_handling.py` | Price panels, CSV loading, synthetic data |
| `adaptive_alpha/factor_generation.py` | Symbolic expressions and candidate generation |
| `adaptive_alpha/testing_evaluation.py` | Robustness tests and factor metrics |
| `adaptive_alpha/experiment_logic.py` | Walk-forward evaluation and budget policies |
| `adaptive_alpha/results_config.py` | Configuration and reports |
| `adaptive_alpha/cli.py` | Command-line interface |
| `tests/` | Unit tests and an end-to-end pipeline check |

## Your own data

Keep proprietary data outside the repository. Supply a local CSV with these required columns:

```text
date,asset,open,high,low,close,volume
```

Optional columns: `shares_outstanding`, `industry`, `exchange`, `share_code`.

```bash
python -m adaptive_alpha.cli build-benchmark \
  --config configs/demo.json \
  --prices /absolute/path/to/private/prices.csv \
  --output artifacts/local
```

Adapt the config's date bounds and window lengths to your dataset. Private data, credentials, and historical experiment outputs are not included. Data files and generated outputs are ignored by Git; review any new file formats before publishing. Results derived from private data should also remain private.

For real-market research, point-in-time metadata, universe membership, delisting returns, and leakage controls require additional validation.

## Tests

```bash
python -m unittest discover -s tests -v
```
