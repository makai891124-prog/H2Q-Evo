# Appendix Template: DAS-GQS Experimental Methods and Results

## A1. Objective and Scope
This appendix documents the reproducible experimental pipeline for the DAS Geometric Quantum Simulator (DAS-GQS), including:
1. Rotor-based reversible evolution in G3.
2. CHSH/Tsirelson validation under geometric correlation.
3. Backend consistency checks (NumPy vs clifford).
4. Statistical convergence with fixed noise conditions.

## A2. Geometric-Physical Mapping
### A2.1 State Representation
Single-particle state is a unit real vector $v \in \mathbb{R}^3$ on the Bloch/Riemann sphere.

### A2.2 Evolution Rule
Rotor:
$$
R = \cos(\theta/2) - B\sin(\theta/2), \quad \|B\|=1
$$
State update:
$$
v' = R v \widetilde{R}
$$
Reversibility check:
$$
\widetilde{R} v' R = v
$$

### A2.3 Correlation and CHSH
Singlet-equivalent geometric correlation:
$$
E(a,b) = -a\cdot b
$$
CHSH expression:
$$
S = E(A,B) - E(A,B') + E(A',B) + E(A',B')
$$
Tsirelson bound target:
$$
|S| \le 2\sqrt{2}
$$

## A3. Noise Model
For axis jitter (Gaussian, std $\sigma$ in radians per side) and independent output flip probability $p$ per side:
$$
E_{\text{noisy}}(\Delta) = -\cos(\Delta)\exp(-\sigma^2)(1-2p)^2
$$
where $\Delta$ is the analyzer angle difference.

## A4. Experimental Setup
### A4.1 Environment
Fill in:
1. OS / CPU / Python version.
2. Package versions (`numpy`, `clifford`, `matplotlib`).
3. Random seed policy.

### A4.2 Core Commands (Reproducible)
```bash
# 1) Core phase demonstration + Tsirelson point check
/Users/imymm/H2Q-Evo/.venv/bin/python h2q_project/das_gqs/chsh_validation.py

# 2) Batch CHSH confidence intervals + noise robustness scenarios
/Users/imymm/H2Q-Evo/.venv/bin/python -m h2q_project.das_gqs.batch_report

# 3) Convergence contrast: analytic vs sampled error over N
/Users/imymm/H2Q-Evo/.venv/bin/python -m h2q_project.das_gqs.convergence_experiment \
  --n-list 200,500,1000,2000,5000,10000,20000 \
  --trials 120 \
  --axis-jitter-deg 2.0 \
  --outcome-flip-prob 0.03 \
  --seed 2026

# 4) Backend consistency unit tests
/Users/imymm/H2Q-Evo/.venv/bin/python -m pytest -q -o addopts='' tests/test_das_gqs_backends.py
```

## A5. Output Artifacts
Expected generated files:
1. `reports/das_gqs_noise_robustness_report.json`
2. `reports/das_gqs_noise_robustness_report.md`
3. `reports/das_gqs_convergence_curve.json`
4. `reports/das_gqs_convergence_curve.csv`
5. `reports/das_gqs_convergence_curve.md`
6. `reports/das_gqs_convergence_curve.png`

## A6. Statistical Analysis Plan
### A6.1 Point and Interval Estimates
For each estimator $\hat{E}$ and $\hat{S}$ report:
1. Mean.
2. Standard error.
3. 95% confidence interval.

### A6.2 Classical Limit Hypothesis Test
At each $N$, test against local-realistic threshold:
$$
H_0: |S| \le 2 \quad \text{vs} \quad H_1: |S| > 2
$$
Report a z-score proxy and two-sided p-value from the empirical uncertainty of $\hat{S}$.

### A6.3 Convergence-Rate Test
Fit:
$$
\log(\text{RMSE}) = \alpha + \beta \log N
$$
Interpretation:
1. $\beta \approx -0.5$: Monte Carlo-like convergence.
2. $R^2$: goodness of log-log scaling.

## A7. Result Tables (Fill Template)
### A7.1 Backend Consistency
| Metric | Value |
|---|---:|
| Max L2 error | [fill] |
| Mean L2 error | [fill] |
| Samples | [fill] |

### A7.2 CHSH and Tsirelson
| Metric | Value |
|---|---:|
| $S$ mean | [fill] |
| 95% CI | [fill] |
| $|S|$ | [fill] |
| $2\sqrt{2}$ | 2.828427 |

### A7.3 Noise Robustness
| Scenario | jitter (deg) | flip prob | $|S|$ | 95% CI excludes 2? |
|---|---:|---:|---:|---|
| ideal | [fill] | [fill] | [fill] | [fill] |
| mild | [fill] | [fill] | [fill] | [fill] |
| moderate | [fill] | [fill] | [fill] | [fill] |
| strong | [fill] | [fill] | [fill] | [fill] |

### A7.4 Convergence
| N | MAE | RMSE | Bias | CI coverage |
|---:|---:|---:|---:|---:|
| [fill] | [fill] | [fill] | [fill] | [fill] |

## A8. Threats to Validity
1. Angle jitter model uses Gaussian approximation in-plane.
2. Outcome-flip model assumes independent Bernoulli noise per side.
3. Finite-trial uncertainty for p-values may be conservative in small trial counts.

## A9. Reproducibility Checklist
1. Fixed random seeds disclosed.
2. Exact commands provided.
3. Raw json/csv artifacts preserved.
4. Plot-generation code included.
5. Backend parity checks documented.
