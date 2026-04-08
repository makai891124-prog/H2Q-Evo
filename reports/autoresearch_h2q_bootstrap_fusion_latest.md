# Autoresearch-H2Q Bootstrap Fusion

- generated_at_utc: `2026-03-10T15:56:44.670449+00:00`
- execute: `True`
- iterations: `1`
- timeout_sec: `900`

## Upstream Autoresearch Summary
- source: `/Users/imymm/H2Q-Evo/external/autoresearch/results.tsv`
- exists: `False`
- keep/discard/crash: `0/0/0`

## Baseline Snapshot
- distill_delta: `1.0`
- research_aggregate: `0.9943158135231227`
- systemic_score: `0.97259375`

## Experiment Ledger
| i | name | status | metric | baseline | after | delta | decision_score | delta_signal | meaning_signal | geometry_signal | axiom_signal | weight_gain_signal | weight_gate | decision_reason |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | distillation_uplift | keep | delta_schema_valid_rate | 1.0 | 1.0 | +0.000000 | 0.5044 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.6943 | 0.1500 | weight gain up (gated) |

## Weight Training Signal
- source: `/Users/imymm/H2Q-Evo/reports/trusted_nano_lora_training_latest.json`
- baseline_loss_improvement_rate: `+0.025638`
- final_loss_improvement_rate: `+0.025638`
- curve:
  - step=0, experiment=baseline, gain=+0.025638, signal=0.694324, gate=0.000000, eff_w=0.000000
  - step=1, experiment=distillation_uplift, gain=+0.025638, signal=0.694324, gate=0.150000, eff_w=0.022500

## MeaningScore
- baseline: `0.937616`
- final: `0.987616`
- curve:
  - step=0, experiment=baseline, score=0.937616
  - step=1, experiment=distillation_uplift, score=0.987616

## Geometric Signal
- baseline_score: `0.542554`
- final_score: `0.542554`
- baseline_boundary_ratio: `0.993133`
- final_boundary_ratio: `0.993133`
- baseline_projection_accel: `-0.022681`
- final_projection_accel: `-0.022681`
- curve:
  - step=0, experiment=baseline, score=0.542554, boundary=0.993133, accel=-0.022681
  - step=1, experiment=distillation_uplift, score=0.542554, boundary=0.993133, accel=-0.022681

## Axiom Consistency
- baseline_score: `0.790913`
- final_score: `0.798913`
- baseline_pass_rate: `0.714286`
- final_pass_rate: `0.714286`
- curve:
  - step=0, experiment=baseline, score=0.790913, pass_rate=0.714286
  - step=1, experiment=distillation_uplift, score=0.798913, pass_rate=0.714286

## Next Bootstrap Plan
- Increase experiment breadth around components marked 'discard' with smaller perturbations.
- Prioritize distillation + consistency coupling before scaling systemic gate complexity.
- Run overnight execute mode with max-iterations >= 12 for true autoresearch-style cadence.
