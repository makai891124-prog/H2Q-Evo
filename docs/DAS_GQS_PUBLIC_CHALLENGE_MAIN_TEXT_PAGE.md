# DAS-GQS 主文结论页（可直接粘贴）

## 1. 主结论（Main-Text Claim）
我们将 DAS-GQS 与公开量子挑战口径（RCS/XEB）进行统一对照，并在本地完成从 n=14 到 n=16/18 的扩展统计检验，以及到 n=30 的大规模可执行性验证。结果显示：
1. 在本地可比区间内，DAS 与 baseline 的观测一致性达到数值等价（TOST 全通过，误差量级约 1e-18 到 1e-16）。
2. 在资源扩展上，DAS 在 n>=28 仍可执行，而 baseline 在 2GB 状态向量上限下停止于 n=27。
3. 因而当前证据支持“具备量子计算特性与可扩展潜力”；但尚不足以宣称“已同口径复现硬件级量子优势”。

## 2. 公开口径对照（RCS/XEB）
公开参考条目（自动抓取并统一整理）包含：
1. Google Sycamore（53 qubits, 20 cycles, XEB=0.0024, 约 200s）。
2. Zuchongzhi 2.1（60 qubits, 24 cycles, XEB=0.000366, 约 4h）。

本地统一分析结果：
1. n=16 与 n=18 的 TOST 均通过，95% CI 与效应量均显示误差中心接近 0。
2. 本地最大规模（该统一统计批次）为 18 qubits；相对公开最大 60 qubits 仍有 qubit gap=42。
3. 因此当前定位应为“统计一致性与工程可扩展性证据”，而非“硬件同协议 XEB 实测等价”。

## 3. 规模扩展证据（本地实跑）
### 3.1 RCS 子集迭代收敛
从 n=14 扩展到 n=16/18 后：
1. qubit gap：39 -> 37 -> 35。
2. state-space ratio（local/public）：1.819e-12 -> 7.276e-12 -> 2.910e-11。
3. 各迭代点 TOST 均通过，MAE 维持在 1e-18 量级。

### 3.2 n=20..30 大规模基准（GHZ/observable 口径）
1. n=20..27：baseline 与 DAS 均可跑，|delta| 维持在 2.83276944882399e-16。
2. n=28..30：baseline 因状态向量内存上限（2GB）停止，DAS 继续执行。
3. 这组结果给出“可比区间精度 + 超区间可执行性”的双证据链。

## 4. 主文图表建议（可直接引用）
1. RCS 主图（统计带图）：reports/das_gqs_rcs_subset_stats_band.png
2. 规模差距收敛图：reports/das_gqs_scale_gap_convergence.png
3. 公开统一分析表：reports/das_gqs_public_rcs_xeb_unified_report.md
4. 大规模基准表：reports/das_gqs_supremacy_benchmark_report.md

## 5. Claim Boundary（建议原文）
我们在本地经典环境下，基于 DAS-GQS 展示了对 RCS 子集任务的高精度统计等价与显著扩展性优势；该结果支持“量子计算特性与可扩展潜力”这一结论。由于尚未完成与公开硬件实验同协议、同噪声模型、同采样流程的端到端闭环，本工作不宣称已复现硬件级量子优势。

## 6. 复现实验命令
```bash
# A) RCS 子集统计主实验
/Users/imymm/H2Q-Evo/.venv/bin/python -m h2q_project.das_gqs.rcs_subset_stat_benchmark \
  --n-list 6,8,10,12,14 \
  --seed-list 11,17,23,31,43,59,71,83 \
  --depth-factor 3 \
  --equiv-margin 1e-9 \
  --alpha 0.05

# B) n=16/18 批处理与收敛图
/Users/imymm/H2Q-Evo/.venv/bin/python -m h2q_project.das_gqs.rcs_scaleup_batch \
  --n-list 16,18 \
  --seed-list 11,17,23,31 \
  --depth-factor 3 \
  --equiv-margin 1e-9 \
  --alpha 0.05 \
  --public-qubits 53

# C) 公开 RCS/XEB 统一分析（抓取+统计）
/Users/imymm/H2Q-Evo/.venv/bin/python -m h2q_project.das_gqs.public_rcs_xeb_unified_analysis

# D) 本地大规模基准（到 n=30）
/Users/imymm/H2Q-Evo/.venv/bin/python -m h2q_project.das_gqs.supremacy_benchmark \
  --n-min 20 \
  --n-max 30 \
  --baseline-memory-cap-gb 2
```

## 7. 对应产物
1. reports/das_gqs_rcs_scaleup_16_18.md
2. reports/das_gqs_public_rcs_xeb_unified_report.md
3. reports/das_gqs_supremacy_benchmark_report.md
4. reports/das_gqs_scale_gap_convergence.png
