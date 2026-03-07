# Research Aggregation Formal Proof Note

- generated_at_utc: `2026-03-07T12:07:34.579606+00:00`
- aggregate_score: `0.978739`
- loo_min_score: `0.971652`
- loo_std_score: `0.008393`
- robust_claim: `True`

## Premises
- P1: Distillation uplift exists and is positive (delta_schema_valid_rate > 0).
- P2: Robustness drift under sessions expansion (30 -> 50) remains bounded.
- P3: Public validation gates pass under baseline and longrun settings.
- P4: Formal logic-closure checks compile in Lean4 and closure facts are true.
- P5: Leave-one-out evidence removal still preserves aggregate score above floor.

## Inference Rules
- R1 (Multi-Evidence Sufficiency): If P1..P4 hold and each evidence family is independent in mechanism, then combined empirical support is stronger than any single metric.
- R2 (Cross-Validation Stability): If LOO min_score remains above acceptance floor and variance is low, claim is not dominated by one artifact.
- R3 (Formal Consistency Filter): If Lean closure compiles with all facts true, the propositional closure over selected gates is logically consistent.

## Conclusion
- Aggregated effect is empirically supported and cross-validated under independent evidence families.

## Threats To Validity
- Construct validity: current aggregate score compresses multiple goals into a weighted scalar; weight choice can bias interpretation.
- Internal validity: shared pipelines may induce correlated errors across metrics (e.g., same upstream data artifacts).
- External validity: results are measured on this project's benchmark/task mix; generalization to new domains is not guaranteed.
- Statistical validity: LOO here is evidence-family ablation, not sample-level bootstrap; uncertainty is partially characterized.

## Counterexample Boundaries
- Boundary B1: If longrun gate flips false while baseline stays true, aggregate claim should be downgraded to conditional robustness.
- Boundary B2: If Lean compile fails or any closure fact becomes false, formal consistency support is invalidated.
- Boundary B3: If sessions expansion causes large drift (e.g., |delta_overall_score| >= 0.02), robustness premise P2 fails.
- Boundary B4: If one evidence family removal drops LOO score below acceptance floor, aggregation becomes single-metric fragile.

## Reproducibility
- Script: `tools/run_research_aggregation_cross_validation.py`
- Artifacts:
  - `/Users/imymm/H2Q-Evo/reports/research_aggregation_cross_validation_latest.json`
  - `/Users/imymm/H2Q-Evo/reports/research_aggregation_cross_validation_latest.md`
  - `/Users/imymm/H2Q-Evo/reports/research_aggregation_cross_validation_proof_note_latest.md`
