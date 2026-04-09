# Research-Architecture Cross Validation

- generated_at_utc: `2026-04-05T14:27:04.618734+00:00`
- sessions: `4`
- total_runs: `12`
- schema_valid_rate: `1.000000`
- overall_score: `0.988955`
- baseline_gate_ok: `True`
- longrun_gate_ok: `True`
- lean_compile_success: `True`

## Aggregate Effectiveness
- score: `0.994316`
- distill_gain: value=1.000000, weight=0.30, note=schema_valid_rate delta from pipeline
- consistency_quality: value=0.988955, weight=0.20, note=overall_score from distilled benchmark
- robustness_30_vs_50: value=0.992249, weight=0.10, note=stability under sessions increase
- public_validation: value=0.989200, weight=0.25, note=baseline/longrun gate and alignment
- formal_closure: value=1.000000, weight=0.15, note=Lean compile + closure facts

## Leave-One-Out Cross Validation
- min_score: `0.991880`
- max_score: `0.996021`
- mean_score: `0.994283`
- std_score: `0.001529`
- left_out=distill_gain, score=0.991880
- left_out=consistency_quality, score=0.995656
- left_out=robustness_30_vs_50, score=0.994545
- left_out=public_validation, score=0.996021
- left_out=formal_closure, score=0.993313

## Proof Argument
- P1: Distillation pipeline has positive schema-valid uplift.
- P2: Robustness from sessions=30 to sessions=50 remains stable.
- P3: Public validation gates pass in baseline and longrun settings.
- P4: Lean4 logical closure compiles with all closure facts true.
- P5: Leave-one-evidence-out aggregate score remains above acceptance floor.
- robust_claim: `True`
- conclusion: Aggregated effect is empirically supported and cross-validated under independent evidence families.

## Paper-to-Module Mapping
- Self-Consistency Improves Chain of Thought Reasoning in Language Models (2022): consistency/robustness -> tools/run_self_model_consistency_benchmark.py, reports/self_model_consistency_distilled_latest.json; https://arxiv.org/abs/2203.11171
- Reflexion: Language Agents with Verbal Reinforcement Learning (2023): self-improvement -> tools/trusted_local_agi_chat.py, tools/collect_self_eval_distill_samples.py, tools/train_self_eval_distillation_adapter.py; https://arxiv.org/abs/2303.11366
- Self-Refine: Iterative Refinement with Self-Feedback (2023): iterative quality uplift -> tools/run_self_eval_distillation_pipeline.py, reports/self_eval_distillation_pipeline_latest.json; https://arxiv.org/abs/2303.17651
- Direct Preference Optimization: Your Language Model is Secretly a Reward Model (2023): alignment stability -> reports/distill_evo_public_validation_latest.json, reports/release_gate_latest.json; https://arxiv.org/abs/2305.18290
- Constitutional AI: Harmlessness from AI Feedback (2022): safety/alignment gates -> tools/run_agi_integrated_validation.py, reports/distill_evo_public_validation_latest.json; https://arxiv.org/abs/2212.08073
- Self-Rewarding Language Models (2024): closed-loop reward improvement -> tools/run_distill_evolution_public_formal_assessment.py, reports/distill_evo_public_formal_assessment_latest.json; https://arxiv.org/abs/2401.10020
- LoRA: Low-Rank Adaptation of Large Language Models (2021): low-cost adaptation -> tools/train_self_eval_distillation_adapter.py, reports/self_eval_distill_model_latest.json; https://arxiv.org/abs/2106.09685
- Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020): factual grounding/provenance -> h2q_project/h2q_server.py, tools/trusted_local_agi_chat.py; https://arxiv.org/abs/2005.11401
- Improving Dialogue Management: Quality Datasets vs Models (2023): data quality sensitivity -> tools/collect_self_eval_distill_samples.py, reports/self_eval_distill_dataset_latest.json; https://arxiv.org/abs/2310.01339
