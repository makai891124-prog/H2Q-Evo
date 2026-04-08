# DAS-GQS 等效计算加速严格可信对比实验报告（2026-03-28）

## 1. 报告目标与边界

本报告针对 H2Q-Evo 中 DAS-GQS 架构，评估其在“等效计算任务”上的加速水平，并给出工程学与学术意义判断。

- 本报告评估的是：在已定义任务与统计口径下，DAS 相对基线状态向量方法的性能/资源收益与结果等效性。
- 本报告不宣称：已在与公开硬件挑战同规模、同协议条件下复现量子优越性。

## 2. 公开可信方法学依据

本报告采用并映射以下公开规则/文献要点：

1. SPEC CPU2017 run/reporting rules
- 强调可重复性、公平对比、完整披露、估算与实测区分、功耗/配置口径规范。

2. MLPerf Inference Rules（inference_rules.adoc）
- 可比性要求：同一系统与框架口径一致。
- 复现性要求：结果可复现。
- 约束要求：禁止 benchmark detection、禁止输入内容投机优化、限制非确定性。
- 统计要求：场景化延迟/吞吐定义与早停统计准则。

3. MLPerf Results Messaging Guidelines
- 非官方审核结果必须明确标注 unverified，不得伪装为官方验证结果。
- 仅允许同口径结果比较，并应明确比较差异。

4. Random Circuit Sampling / XEB 公共参考
- Google 53 qubit, 20 cycles；Zuchongzhi 60 qubit, 24 cycles 等公开 RCS/XEB 参考点用于规模边界对照。

## 3. 本地实验数据来源（可复验）

- `reports/das_gqs_supremacy_benchmark_report.json`
- `reports/das_gqs_rcs_subset_stats.json`
- `reports/das_gqs_public_rcs_xeb_unified_report.json`

## 4. 对比设计

### 4.1 公平性与一致性

- 基线：状态向量方法（内存复杂度 O(2^n)）。
- 对照：DAS 方法（报告中给出的链式 GHZ 结构下近似 O(n) 内存）。
- 对比维度：
  - 时间：`speedup = baseline_time / das_time`
  - 内存：`compression = baseline_bytes / das_bytes`
  - 等效性：TOST（alpha=0.05，margin=1e-9）+ CI95 + RMSE/MAE

### 4.2 可重复性信息

- RCS 子集统计：n in {6,8,10,12,14}，8 个 seed，每 n 120 样本。
- 公开口径统一分析：本地 n in {16,18}，seed {11,17,23,31}。
- Supremacy scaling：n=20..30，基线内存上限 2 GiB。

## 5. 核心结果

### 5.1 性能与资源

在基线可运行重叠区间 n=20..27：

- 加速倍数范围：374.55x 到 117879.92x
- 几何平均加速：15489.76x
- 中位近似加速：23133.82x
- 内存压缩范围：17772.47x 到 1677721.60x
- 几何平均内存压缩：171538.97x

规模边界（本次配置下）：

- 基线最大可运行：27 qubits
- DAS 已测试到：30 qubits
- 在状态空间规模上对应提升：2^(30-27)=8x

### 5.2 结果等效性

RCS 子集统计（n=6..14）显示：

- TOST 通过率：100%（5/5）
- MAE 范围：2.716e-18 到 2.883e-16
- 全部显著低于预设等效边界 1e-9

统一公开口径对照（本地 n=16,18）显示：

- TOST 全通过
- 效应量（cohen_d, hedges_g）为 0（报告记录）
- xeb_proxy 约 0.9999999975（用于本地一致性代理）

### 5.3 与公开硬件挑战的规模差距

由 `das_gqs_public_rcs_xeb_unified_report.json`：

- 公开参考最大规模：60 qubits
- 本地最大规模：18 qubits（该统一分析项）
- qubit gap：42

结论：本地统计一致性很强，但与公开硬件挑战仍非同规模同条件，不能据此声称硬件级量子优势复现。

## 6. 工程学意义

1. 资源效率意义明确
- 在等效任务定义下，DAS 相对状态向量法实现数量级时间与内存收益。

2. 可部署性潜力
- 在固定内存上限下，将可运行问题规模从 27 扩展到 30 qubits（对应 8x 状态空间），对资源受限环境具有现实价值。

3. 评测治理意义
- 采用公开规则映射（可重复、可披露、口径一致）后，结果可用于工程决策而非仅内部 demo。

## 7. 学术意义

1. 等效计算主张获得统计支持
- 在多 n、多 seed、多样本条件下，TOST + CI + 误差指标共同支持“数值等效”。

2. 方法学可迁移
- 将公开 benchmark 治理原则引入本地评估，形成“算法创新 + 评测合规”联合范式。

3. 外推边界清晰
- 当前证据支持“等效计算加速”而非“公开硬件挑战同口径胜出”。该边界本身提升了学术论证可信度。

## 8. 可信声明（严格口径）

- 本报告全部数据来自本地复现实验与公开规则文本映射。
- 本报告不是 MLCommons 官方审计/验证结果。
- 若对外发布，建议遵循 MLPerf messaging 规范显式标注：Result not verified by MLCommons Association.

## 9. 最终判定

在当前可复验数据与公开方法学映射下，DAS-GQS 的“等效计算加速”结论为：

- 可证据支持：成立（强）
- 工程意义：高
- 学术意义：中高（在当前规模边界内）
- 禁止外推：不得将该结论表述为已在公开硬件同口径挑战中实现量子优越性复现

## 10. 具体算法开源披露（完整）

本报告对应算法与评测实现已在仓库源码中公开，可直接审阅、复跑、复算。

### 10.1 算法主链路（DAS 采样与计算计划）

1. 线程-批大小联合计算计划
- 入口函数：`resolve_batch_sampling_compute_plan`
- 作用：基于硬件画像、候选线程集合、matmul 探针结果生成可复用 compute plan，并写入/读取离线缓存。

2. CHSH 批估计
- 入口函数：`estimate_chsh_batch`
- 作用：对给定噪声配置与采样参数进行批量估计，输出期望值、误差与计划元数据。

3. 噪声鲁棒报告
- 入口函数：`noise_robustness_report`
- 作用：遍历噪声网格，输出鲁棒性统计，并在结果中携带硬件与计算计划信息。

### 10.2 报告与评测脚本

1. 批报告生成
- 脚本：`h2q_project/das_gqs/batch_report.py`
- 关键入口：`parse_args`、`main`
- 关键行为：调用 `resolve_batch_sampling_compute_plan`，打印并落盘 cache hit/miss、threads、统计结果。

2. 等效统计检验
- 脚本：`h2q_project/das_gqs/rcs_subset_stat_benchmark.py`
- 关键算法：`_tost_equivalence`
- 输出：MAE/RMSE/CI95/TOST p 值/是否通过等效。

3. 公开口径统一分析
- 脚本：`h2q_project/das_gqs/public_rcs_xeb_unified_analysis.py`
- 关键模块：`fetch_public_rcs_xeb_table`、`build_gap_summary`
- 输出：本地统计与公开 RCS/XEB 对照、规模 gap 与谨慎结论。

4. 规模扩展基准
- 脚本：`h2q_project/das_gqs/supremacy_benchmark.py`
- 关键参数：`baseline_memory_cap_bytes`
- 输出：基线可运行边界、DAS 可运行范围与误差差值。

### 10.3 开源透明性声明

- 未使用闭源二进制黑盒评测器。
- 关键统计与对比逻辑均在仓库 Python 代码中可见。
- 报告 JSON/MD 与脚本输入参数可一一对应，便于第三方复验。

## 11. 现有边界（详细）

1. 任务边界
- 当前“等效加速”结论针对已定义任务集（CHSH、噪声鲁棒、RCS 子集、统一 XEB 代理分析）。
- 结论不自动外推到任意量子算法或任意电路分布。

2. 规模边界
- 本地统一公开口径分析最大到 n=18；supremacy 脚本中 DAS 测到 n=30，基线受 2 GiB 上限在 n=27 截止。
- 与公开 53/60 qubit 挑战存在显著规模差距。

3. 指标边界
- 本地 `xeb_proxy` 是一致性代理指标，不等同于公开硬件实验中的完整 XEB 统计流程。
- 结果可用于“本地等效性与效率”判断，不可替代官方挑战成绩。

4. 合规边界
- 本报告为非 MLCommons 官方审计结果；对外必须显式标注 unverified。
- 不允许将本地 derived metric 叙事伪装为官方验证分数。

5. 工程边界
- 线程/缓存计划依赖当前硬件与运行时；跨平台迁移需重新探针与缓存再生成。
- 部分缓存命中收益依赖 workload 重复性与参数稳定性。

## 12. 复现实验命令（建议）

在仓库根目录执行（确保 `PYTHONPATH=.`）：

```bash
PYTHONPATH=. python3 h2q_project/das_gqs/supremacy_benchmark.py
PYTHONPATH=. python3 h2q_project/das_gqs/rcs_subset_stat_benchmark.py
PYTHONPATH=. python3 h2q_project/das_gqs/public_rcs_xeb_unified_analysis.py
PYTHONPATH=. python3 h2q_project/das_gqs/batch_report.py --autotune-threads --compute-plan-cache reports/das_batch_compute_plan_cache.json
```

建议至少连续运行两次 `batch_report.py`，验证首轮 cache miss、次轮 cache hit 的可复现行为。

