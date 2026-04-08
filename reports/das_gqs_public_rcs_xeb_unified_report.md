# DAS-GQS Unified Public RCS/XEB Analysis

- timestamp (UTC): 2026-03-27T16:33:46.031054+00:00
- source status: {'wiki_xeb': 'ok', 'google_rcs': 'ok'}

## Public RCS/XEB Table
| source | platform | qubits | cycles | XEB | runtime(s) | samples |
|---|---|---:|---:|---:|---:|---:|
| wiki | Google Sycamore (2019) | 53 | 20 | 0.002400 | 200.0 | 1000000 |
| wiki | Zuchongzhi 2.1 (2021) | 60 | 24 | 0.000366 | 14400.0 |  |

## Local Unified Statistical Tests
| n | depth | samples | MAE | 95%CI(MAE) | RMSE | max | TOST p1 | TOST p2 | pass | cohen_d | hedges_g | XEB_proxy |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---:|---:|---:|
| 16 | 48 | 60 | 2.530e-18 | [1.132e-18, 3.927e-18] | 6.031e-18 | 3.226e-17 | 0.000e+00 | 0.000e+00 | True | 0.000e+00 | 0.000e+00 | 1.000000 |
| 18 | 54 | 60 | 2.416e-18 | [7.989e-19, 4.032e-18] | 6.781e-18 | 2.844e-17 | 0.000e+00 | 0.000e+00 | True | 0.000e+00 | 0.000e+00 | 1.000000 |

## Gap Summary
- public_max_qubits: 60
- local_max_qubits: 18
- qubit_gap: 42
- state_space_ratio_local_over_public: 2.273737e-13
- public_sycamore_xeb: 2.400000e-03
- public_min_xeb: 3.660000e-04
- local_mean_xeb_proxy: 1.000000e+00
- local_proxy_over_sycamore_xeb: 4.166667e+02
- local_tost_pass_rate: 1.000000e+00

## Verdict
- 本地 DAS 在 n=16/18 上通过统一统计检验（TOST+置信区间+效应量），并保持高一致性；但与公开 RCS/XEB 挑战仍存在规模差距，应表述为‘具备量子计算特性与可扩展潜力’，而非‘已在同口径公开挑战上完成硬件级量子优势复现’。

## Notes
- XEB_proxy is a bounded local consistency indicator derived from DAS-vs-baseline error scale.
- It is not a hardware-measured linear XEB and must not be used as a direct replacement claim.
