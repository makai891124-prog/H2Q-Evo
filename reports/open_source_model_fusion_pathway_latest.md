# Open-Source Model Fusion Pathway

- generated_at_utc: `2026-04-05T13:58:38.653076+00:00`
- discovered_models: `12`
- ollama_scan_ok: `True`

## Tier Maturity
- edge: `stable`
- bridge: `stable`
- core: `developing`

## Dynamic Separation
| model | size_b | tier | edge_score | core_score | specialties |
|---|---:|---|---:|---:|---|
| test_model | 0.00 | bridge | 0.45 | 0.45 | general |
| deepseek-coder-v2-236b | 236.00 | core | 0.27 | 0.97 | coding, general_reasoning |
| deepseek-coder-v2-236b-compressed:latest | 236.00 | core | 0.27 | 0.97 | coding, general_reasoning |
| deepseek-coder-v2:236b | 236.00 | core | 0.27 | 0.97 | coding, general_reasoning |
| deepseek-coder:33b | 33.00 | core | 0.47 | 0.82 | coding, general_reasoning |
| codellama-7b | 7.00 | edge | 0.89 | 0.52 | coding, general_reasoning |
| deepseek-coder-6.7b | 6.70 | edge | 0.89 | 0.52 | coding, general_reasoning |
| deepseek-coder:6.7b | 6.70 | edge | 0.89 | 0.52 | coding, general_reasoning |
| llama-2-7b-chat | 7.00 | edge | 0.87 | 0.52 | instruction, general_reasoning |
| llama2:7b | 7.00 | edge | 0.84 | 0.52 | general_reasoning |
| qwen2.5-0.5b | 0.50 | edge | 0.92 | 0.32 | general_reasoning |
| tinyllama-1.1b | 1.10 | edge | 0.92 | 0.32 | general_reasoning |

## Edge -> Core Self-Improvement Process
- S1_edge_skill_hardening: Harden fast, low-cost capabilities (schema fidelity, formatting, retrieval snippets)
  - entry: `edge_schema_valid_rate=1.0`
  - target: `>= 0.80`
  - exit_gate: `delta_schema_valid_rate > 0 and schema_valid_rate >= 0.80`
  - drivers: `tools/run_self_eval_distillation_pipeline.py, tools/run_self_model_consistency_benchmark.py`
  - candidate_models: `codellama-7b, deepseek-coder-6.7b, deepseek-coder:6.7b, llama-2-7b-chat`
- S2_bridge_transfer_alignment: Transfer edge gains into robust multi-turn self-evaluation and adapter compatibility
  - entry: `bridge_consistency_overall_score=0.9889546351084813`
  - target: `>= 0.82`
  - exit_gate: `overall_score >= 0.82 and semantic_consistency >= 0.80`
  - drivers: `tools/openclaw_h2q_adapter.py, tools/run_self_model_consistency_benchmark.py`
  - candidate_models: `test_model`
- S3_core_reasoning_consolidation: Consolidate long-horizon planning, cross-validation stability, and release-gate trust
  - entry: `core_systemic_score=0.0`
  - target: `>= 0.85`
  - exit_gate: `aggregate.score >= 0.85 and cv.min_score >= 0.80`
  - drivers: `tools/dynamic_blueprint_bootstrap.py, tools/run_systemic_platform_joint_capability_assessment.py`
  - candidate_models: `deepseek-coder-v2-236b, deepseek-coder-v2-236b-compressed:latest, deepseek-coder-v2:236b, deepseek-coder:33b`
- S4_formalized_self_improvement: Convert empirical gains into formal closure and repeatable governance checks
  - entry: `formal_logic_closed=False`
  - target: `true`
  - exit_gate: `logic_closure all_true and gate_ok true`
  - drivers: `tools/run_distill_evolution_public_formal_assessment.py, tools/release_gate.py`
  - candidate_models: `deepseek-coder-v2-236b, deepseek-coder-v2-236b-compressed:latest, test_model`

## Capability Snapshot
- edge_schema_valid_rate: `1.0`
- edge_delta_schema_valid_rate: `1.0`
- bridge_consistency_overall_score: `0.9889546351084813`
- core_systemic_score: `0.0`
- core_cv_min_score: `0.9545454545454545`
- formal_logic_closed: `False`
