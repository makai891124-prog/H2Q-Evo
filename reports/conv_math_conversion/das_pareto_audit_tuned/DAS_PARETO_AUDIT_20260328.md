# DAS Pareto Audit (Multi-seed + Multi-rank)

- model: `distilgpt2`
- ranks: `[32, 64]`
- seeds: `[11, 22, 33]`
- distill hparams: `temp=1.35->0.12, topk=10, rank_w=0.75, mse_w=0.03, margin=0.08`

## Aggregated Metrics

- rank 32: cosine=0.9999, top5=0.5325, speedup=1.7095x, compression=17.4468x
- rank 64: cosine=0.9999, top5=0.6664, speedup=1.6021x, compression=8.7238x

## Pareto Front

- rank 32: cosine=0.9999, speedup=1.7095x, compression=17.4468x
- rank 64: cosine=0.9999, speedup=1.6021x, compression=8.7238x

JSON report: `reports/conv_math_conversion/das_pareto_audit_tuned/das_pareto_audit_20260328.json`