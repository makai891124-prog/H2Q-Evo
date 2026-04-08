# DAS Pareto Audit (Multi-seed + Multi-rank)

- model: `distilgpt2`
- ranks: `[32]`
- seeds: `[11, 22, 33]`
- distill hparams: `temp=1.35->0.12, topk=10, rank_w=0.75, mse_w=0.03, margin=0.08, hard_k=12, hard_w=0.45, split=0.40, stage1=0.25`

## Aggregated Metrics

- rank 32: cosine=0.9997, top5=0.6868, speedup=1.7226x, compression=17.4468x

## Pareto Front

- rank 32: cosine=0.9997, speedup=1.7226x, compression=17.4468x

JSON report: `reports/conv_math_conversion/das_pareto_audit_rank32_hardneg/das_pareto_audit_20260328.json`