# DAS-GQS Convergence Contrast Experiment

## Setup
- axis jitter (deg): 2.0
- outcome flip probability: 0.03
- N list: [200, 500]
- trials per N: 8
- selected torch threads: 2
- compute-plan cache hit: True

## Convergence Fit
- slope (log RMSE vs log N): -0.269635
- intercept: -1.173559
- R^2: 1.000000
- ideal Monte Carlo convergence reference: slope = -0.5

## Results
| N | S_theory | S_mean | Bias | MAE | RMSE | CI coverage | z(|S|-2) | p(two-sided) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 200 | -2.496155 | -2.498750 | -0.002595 | 0.063750 | 0.074113 | 1.000 | 17.816 | 5.324e-71 |
| 500 | -2.496155 | -2.499000 | -0.002845 | 0.048039 | 0.057889 | 1.000 | 22.834 | 2.109e-115 |
