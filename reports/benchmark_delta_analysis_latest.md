# Benchmark Delta Analysis

- generated_at_utc: 2026-03-09T14:00:16.033289+00:00
- after: `/Users/imymm/H2Q-Evo/agi_benchmark_results.json`
- before: `/Users/imymm/H2Q-Evo/reports/agi_benchmark_results_before_latest.json`
- overall_before: 0.378932
- overall_after: 0.378932
- overall_delta: +0.000000

## Top Regressions
| Task | Before | After | Delta |
|---|---:|---:|---:|

## Top Improvements
| Task | Before | After | Delta |
|---|---:|---:|---:|

## Failure Signals
- model-json-guard.parse_errors: 701

## Potential Causes
- High JSON parse-error debt suggests response formatting drift and contract leakage.

## Recommended Attempts
- Force /generate contract mode on probe traffic and verify with a dedicated contract smoke test in CI.
- Add per-task calibration constants (NLI/Bool/MCQ) and tune on a held-out split before full benchmark runs.
- Track per-task confidence histograms and reject low-confidence predictions with deterministic fallback labels.
- Gate release on three checks: infra_valid=true, parse_errors<=target, and non-negative overall_score delta.
