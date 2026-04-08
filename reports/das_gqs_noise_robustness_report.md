# DAS-GQS Batch Statistical Report

## 1. Backend Consistency
- backends: ['numpy', 'clifford']
- samples: 32
- max L2: 2.719480e-16
- mean L2: 1.177222e-16

## 2. CHSH Batch Estimate (95% CI)
- S mean: -2.828533
- S 95% CI: [-2.844536, -2.812530]
- |S|: 2.828533
- Tsirelson target: 2.828427
- selected threads: 2
- compute plan cache hit: False

## 3. Noise Robustness
| scenario | jitter(deg) | flip_p | S mean | 95% CI low | 95% CI high | |S| | CI excludes 2? |
|---|---:|---:|---:|---:|---:|---:|---|
| ideal | 0.00 | 0.000 | -2.828000 | -2.844006 | -2.811994 | 2.828000 | True |
| mild | 1.00 | 0.020 | -2.597667 | -2.614874 | -2.580459 | 2.597667 | True |
| moderate | 3.00 | 0.050 | -2.179867 | -2.198835 | -2.160898 | 2.179867 | True |
| strong | 5.00 | 0.100 | -1.758667 | -1.778993 | -1.738341 | 1.758667 | False |
