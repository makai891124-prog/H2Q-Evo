# DAS Pareto Audit (Multi-seed + Multi-rank)

- model: `HuggingFaceTB/SmolLM2-360M`
- ranks: `[32]`
- seeds: `[11, 22, 33]`
- distill hparams: `temp=1.35->0.12, topk=10, rank_w=0.75, mse_w=0.03, margin=0.08, hard_k=12, hard_w=0.45, split=0.40, stage1=0.25`

## Aggregated Metrics

- rank 32: cosine=0.9701, top5=0.5732, speedup=1.8697x, compression=20.4170x

## Pareto Front

- rank 32: cosine=0.9701, speedup=1.8697x, compression=20.4170x

JSON report: `reports/conv_math_conversion/sota_smallmodel_smollm2_360m/pareto_audit/das_pareto_audit_20260328.json`