# Distill-Evolution Public Formal Assessment

- generated_at_utc: `2026-03-07T04:49:38.083998+00:00`

## Distillation Capability
- sessions: `50`
- total_runs: `150`
- schema_valid_rate: `1.000000`
- overall_score: `0.913007`
- grade: `A`
- pipeline_delta_schema_valid_rate: `+1.000000`

## Robustness (30 vs 50)
- sessions=30 schema_valid_rate: `1.000000`
- sessions=30 overall_score: `0.913782`
- sessions=50 schema_valid_rate: `1.000000`
- sessions=50 overall_score: `0.913007`
- delta_schema_valid_rate(50-30): `+0.000000`
- delta_overall_score(50-30): `-0.000775`

## Public Validation (Open Experimental Set)
- baseline_gate_ok: `True`
- longrun_gate_ok: `True`
- baseline_alignment_overall: `0.999817`
- longrun_alignment_overall: `0.956983`
- baseline_blueprint_ok_rate: `0.800000`
- longrun_blueprint_ok_rate: `1.000000`

## Lean4 Logical Closure
- lean_file: `/Users/imymm/H2Q-Evo/reports/distill_evolution_logic_closure_1772858977.lean`
- lean_compile_success: `True`
- facts: `{'distill_pipeline_all_steps_ok': True, 'public_validation_all_steps_ok': True, 'distilled_schema_positive': True, 'baseline_gate_ok': True, 'longrun_gate_ok': True}`
