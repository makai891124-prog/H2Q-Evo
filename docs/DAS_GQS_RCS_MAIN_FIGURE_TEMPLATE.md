# Main-Text Figure (Completed): RCS Subset Consistency and Statistical Equivalence

## Figure X. Multi-Observable Consistency on RCS Subset
We evaluate a random Clifford-circuit subset (RCS subset) and compare baseline state-vector simulation with DAS lazy Heisenberg projection across single- and two-qubit Pauli observables. The solid line is mean absolute error (MAE), shaded region is its 95% confidence band, and the dashed horizontal line is the pre-registered equivalence margin $\epsilon=10^{-9}$. A TOST procedure is applied per-$n$ to assess statistical equivalence. Across $n\in\{6,8,10,12,14\}$ and 8 random seeds, all per-$n$ TOST tests pass.

Figure path:
- [reports/das_gqs_rcs_subset_stats_band.png](reports/das_gqs_rcs_subset_stats_band.png)

## Methods (Main Text Short Form)
1. Circuit family: random subset of Clifford gates {H, S, Sdg, CNOT}.
2. Depth rule: $d = 3n$.
3. Observables: single-qubit {X,Y,Z} and two-qubit {XX,YY,ZZ} on fixed local neighborhoods.
4. Seeds: {11, 17, 23, 31, 43, 59, 71, 83}.
5. Statistical protocol:
- Primary metric: MAE between DAS and baseline expectations.
- Interval: 95% CI for MAE.
- Equivalence test: TOST on signed error with margin $\epsilon=10^{-9}$ and significance $\alpha=0.05$.

## Result Table (Filled)
| n | depth | samples | MAE | 95% CI | RMSE | max | TOST p1 | TOST p2 | pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 6 | 18 | 120 | 6.738e-17 | [-2.649e-17, 1.612e-16] | 5.268e-16 | 4.996e-15 | 0.000e+00 | 0.000e+00 | True |
| 8 | 24 | 120 | 1.379e-16 | [-4.861e-17, 3.244e-16] | 1.047e-15 | 8.438e-15 | 0.000e+00 | 0.000e+00 | True |
| 10 | 30 | 120 | 9.867e-17 | [-8.988e-17, 2.872e-16] | 1.054e-15 | 1.155e-14 | 0.000e+00 | 0.000e+00 | True |
| 12 | 36 | 120 | 2.883e-16 | [-1.263e-16, 7.028e-16] | 2.325e-15 | 2.276e-14 | 0.000e+00 | 0.000e+00 | True |
| 14 | 42 | 120 | 2.716e-18 | [1.543e-18, 3.890e-18] | 7.075e-18 | 3.152e-17 | 0.000e+00 | 0.000e+00 | True |

## Cross-Seed Robustness (Filled)
| seed | samples | MAE | RMSE | max | pass(max<=eps) |
|---:|---:|---:|---:|---:|---|
| 11 | 75 | 2.540e-18 | 7.144e-18 | 3.152e-17 | True |
| 17 | 75 | 3.058e-16 | 2.628e-15 | 2.276e-14 | True |
| 23 | 75 | 1.166e-16 | 9.743e-16 | 8.438e-15 | True |
| 31 | 75 | 4.065e-17 | 3.334e-16 | 2.887e-15 | True |
| 43 | 75 | 1.727e-16 | 1.067e-15 | 7.772e-15 | True |
| 59 | 75 | 3.201e-18 | 7.518e-18 | 3.469e-17 | True |
| 71 | 75 | 1.538e-16 | 1.320e-15 | 1.144e-14 | True |
| 83 | 75 | 1.567e-16 | 1.333e-15 | 1.155e-14 | True |

## Authenticity Check: Real Experiment vs Hardcoded Output
结论：本次结果属于真实数值实验，不是硬编码静态表。

证据链：
1. 实验数据由 [h2q_project/das_gqs/rcs_subset_stat_benchmark.py](h2q_project/das_gqs/rcs_subset_stat_benchmark.py) 动态生成。
2. 电路是按 seed 随机采样（`np.random.default_rng(seed)`），不是固定常量数组。
3. Baseline 路径执行了实际 state-vector 演化（H/S/SDG/CNOT 顺序作用）。
4. DAS 路径执行了实际 Heisenberg 反向传播投影，不展开全局张量。
5. 误差、CI、TOST 由运行时统计计算生成，并导出到结果文件：
   - [reports/das_gqs_rcs_subset_stats.json](reports/das_gqs_rcs_subset_stats.json)
   - [reports/das_gqs_rcs_subset_stats.csv](reports/das_gqs_rcs_subset_stats.csv)
   - [reports/das_gqs_rcs_subset_stats.md](reports/das_gqs_rcs_subset_stats.md)
6. 本轮终端记录显示命令已执行成功并产出上述文件。

边界说明：
1. 这是经典仿真上的对照实验，不是量子硬件实机采样。
2. 电路族为 RCS 子集（Clifford-like），不是通用随机通用门全集。
3. 因此该结果支持“方法在该任务域内的一致性与效率”，不等于对所有量子电路的一步到位普适证明。

## 学术与工程意义
### 学术意义
1. 在可控 RCS 子集上给出“多观测+多种子+等价检验”的严格证据链，超越单点数值对齐。
2. 将 DAS 的“图路径懒投影”从概念叙事推进到可检验统计框架（TOST + CI + robustness）。
3. 为后续非 Clifford 扩展与复杂度理论分析提供可复现实验基线。

### 工程意义
1. 在不展开 2^n 张量的前提下完成多观测一致性验证，显示明显内存可扩展性优势。
2. 统计报告、CSV、主文图自动化产出，减少论文复现实验的人工作图与抄表风险。
3. 形成“脚本->数据->图->主文段落”的闭环流水线，可直接接入 CI 与版本化评审。

## Reproducibility Command
```bash
/Users/imymm/H2Q-Evo/.venv/bin/python -m h2q_project.das_gqs.rcs_subset_stat_benchmark \
  --n-list 6,8,10,12,14 \
  --seed-list 11,17,23,31,43,59,71,83 \
  --depth-factor 3 \
  --equiv-margin 1e-9 \
  --alpha 0.05
```

## Artifact Paths
1. [reports/das_gqs_rcs_subset_stats.json](reports/das_gqs_rcs_subset_stats.json)
2. [reports/das_gqs_rcs_subset_stats.csv](reports/das_gqs_rcs_subset_stats.csv)
3. [reports/das_gqs_rcs_subset_stats.md](reports/das_gqs_rcs_subset_stats.md)
4. [reports/das_gqs_rcs_subset_stats_band.png](reports/das_gqs_rcs_subset_stats_band.png)

## Main-Text Integration Note
For a camera-ready conclusion page that integrates:
1. public RCS/XEB unified gap analysis,
2. n=16/18 scale-up statistical evidence,
3. local large-scale run (up to n=30),

see:
- [docs/DAS_GQS_PUBLIC_CHALLENGE_MAIN_TEXT_PAGE.md](docs/DAS_GQS_PUBLIC_CHALLENGE_MAIN_TEXT_PAGE.md)
