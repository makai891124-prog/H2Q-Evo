# DeepSeek-Assisted Transition Plan (Teacher -> Local Student)

- generated_at_utc: `2026-03-06T17:41:35Z`
- objective: `Use external DeepSeek assistance for short-term capability gain, then distill into a fully local composite architecture with measurable anti-pseudo-pass guarantees.`

## Verified Evidence Snapshot

Source artifacts:
- `reports/deepseek_assisted_bootstrap_transition_latest.json`
- `reports/deepseek_assisted_release_gate_1772818796.json`
- `reports/trusted_local_agi_chat_session_1772818268.json`

Observed metrics:
- bootstrap_overall_ok: `true`
- bootstrap_cycles: `2`
- bootstrap_strategy_ok_rate: `1.0`
- release_gate_ok: `true`
- assist_provider: `deepseek`
- assist_calls / success_calls: `6 / 6`
- assist_success_rate: `1.0`
- breadth / horizon / robustness: `1.0 / 1.0 / 1.0`
- interactive_success_rate: `1.0` (`24` tasks)
- trust_score (self-eval probe session): `0.920896`
- hard_fail_triggered_on_invalid_schema: `true` (non-zero termination behavior already observed)

Interpretation:
- External-assist path is operational and stable under current gate profile.
- Hard-fail behavior is now available to block invalid self-eval JSON from being counted as success.
- Remaining bottleneck is not trust/gate score, but conversion of introspection prompts into valid, concrete JSON at high reliability.

## Reliability And Complexity Assessment

Reliability status:
- Gate-level reliability: high (bootstrap + release gate both pass).
- Structured self-eval reliability: still fragile in sampled session (fallback output observed, schema invalid -> hard fail).
- Anti-pseudo-pass guardrail: significantly improved (invalid JSON no longer silently accepted when hard-fail is enabled).

Complexity model:
- Let `C` = bootstrap cycles, `A` = max actions per cycle, `G` = release-gate orchestration steps.
- Controller-level bound per run: `O(C*A + G)`.
- Current concrete run used `C=2`, `A<=2`, `G=5` major steps (`trusted_center`, `daemon_round`, `monitor_snapshot`, `unified_framework`, `capability_registry`).
- Distillation overhead (new): `O(N*T)` where `N` is curated teacher traces and `T` is local fine-tune/adapter updates.

## Transition Architecture (Causal/Light-Cone Compatible)

Phase 0: Stabilize collection contracts
- Keep strict JSON attempt before fallback for self-eval prompts.
- Keep `--self-eval-hard-fail-on-invalid` enabled in benchmark/probe pipelines.
- Persist per-attempt route + schema diagnostics for every introspection sample.

Phase 1: Teacher trace harvesting (DeepSeek)
- Collect high-quality trajectories only when all gates pass:
: strict schema valid + no placeholder tokens + executable action/metric wording.
- Store paired records:
: `prompt`, `teacher_json`, `route_meta`, `trust_meta`, `post-check result`.

Phase 2: Local student distillation
- Distill into local composite modules (reasoning + planner + verifier) with causal truncation:
: each response head sees only allowed prior context slices (light-cone boundary), preventing leakage shortcuts.
- Train objective mix:
: schema fidelity loss + action concreteness loss + metric verifiability loss + route-avoidance penalty for fallback dominance.

Phase 3: Progressive de-externalization
- Reduce DeepSeek call budget in stages (`100% -> 50% -> 20% -> 0%`) only if all gates remain green for a fixed window.
- Fail-safe rollback rule:
: if schema-valid rate or hard-fail incidence regresses beyond threshold, restore previous assistance tier.

## Concrete Exit Criteria

To declare "mostly local" readiness:
- self_eval_schema_valid_rate >= `0.90` on rolling 100 introspection prompts
- placeholder_acceptance_rate == `0.0`
- hard_fail_rate <= `0.05` after repair attempts
- fallback_route_ratio <= `0.20` for self-eval/self-programming prompts
- release_gate_ok == `true` for `>= 5` consecutive assisted-decrease checkpoints

## Instrumentation To Add Next

1. Add transition KPIs into one-click summary output
- `self_eval_schema_valid_rate`
- `hard_fail_count`
- `strict_json_attempt_count`
- `fallback_ratio_self_eval`
- `teacher_assist_dependency_ratio`

2. Add per-round distillation ledger
- accepted teacher traces
- rejected traces with reason codes
- local-student agreement rate vs teacher JSON schema fields

3. Add regression alarms
- Trigger warning if fallback ratio increases by `>0.1` over 3 rounds.
- Trigger hard gate if hard-fail count spikes above configured ceiling.

## Bottom Line

DeepSeek-assisted bootstrapping is now a validated capability-amplification path under your current gates. The architecture is ready for the next step: convert the external teacher advantage into local student reliability with hard-fail-backed observability, then phase out external dependence with explicit rollback controls.
