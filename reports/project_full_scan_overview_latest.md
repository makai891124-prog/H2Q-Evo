# H2Q-Evo 项目全景扫描报告（整体把握版）

- 生成时间（UTC）: `2026-03-07`
- 扫描范围: 仓库 `H2Q-Evo` 全量目录（结构、入口、子系统、工件、流程）
- 报告目的: 帮助你快速建立“代码层 + 运行层 + 证据层”的整体认知框架

## 1. 项目一句话画像
H2Q-Evo 是一个以“自演化调度 + 多模块验证 + 报告工件驱动”为核心形态的大型实验仓库：既包含模型/算法实验代码，也包含高密度自动化评估、发布门禁、监控与形式化验证链路。

## 2. 规模概况（按当前仓库扫描）
以下统计已排除常见虚拟环境与缓存目录（如 `.venv`、`.git`、`node_modules` 等）:

- Python 文件: `2148`
- Markdown 文件: `1060`
- JSON 文件: `3711`
- `reports/` 工件文件: `7156`
- 顶层条目数量: `828`

补充局部统计:
- `tools/*.py`: `48`
- `h2q_project/**/*.py`: `1443`
- 根目录 `*.py`: `325`
- 测试类脚本（`*test*.py`）: `149`
- `reports/*_latest.json`: `60`
- `reports/*_latest.md`: `52`

解读:
- 这是“代码 + 实验记录 +报告产线”三层并重的仓库，不是单一模型项目。
- 真正的复杂度不仅来自源码，还来自运行工件与流程耦合。

## 3. 顶层结构（职责分区）
关键目录（节选）:
- `h2q_project/`: 主体实现（核心算法、服务、训练/实验、模块化架构）
- `tools/`: 自动化工具链（门禁、监控、验证、审计、报告生成）
- `reports/`: 高密度运行工件（JSON/MD/PNG/latest 快照）
- `tests/`: 测试与聚焦验证
- `models/`, `checkpoints/`, `training_checkpoints/`: 模型与训练产物
- `benchmarks/`: 基准任务与评测输入
- `docs/` 与根目录大量 `.md`: 历史报告、说明、总结

## 4. 核心入口与主运行链路

### 4.1 顶层调度入口
- `evolution_system.py`
  - 角色: 全局演化调度器（状态加载、循环调度、可选 Docker、可选 API）
  - 状态文件: `evo_state.json`, `project_memory.json`
  - 运行日志: `evolution.log`

### 4.2 服务入口
- `h2q_project/h2q_server.py`
  - 角色: FastAPI 推理服务层（`/chat`, `/generate` 等）
  - 与数学/决策模块对接: DAS Core、DDE、SST、Holomorphic Middleware

### 4.3 一条典型“实验到结论”路径
1. 采样/训练/蒸馏: `tools/collect_self_eval_distill_samples.py` -> `tools/train_self_eval_distillation_adapter.py` -> `tools/run_self_eval_distillation_pipeline.py`
2. 一致性基准: `tools/run_self_model_consistency_benchmark.py`
3. 集成验证: `tools/run_agi_integrated_validation.py`
4. 公开验证 + 长跑指标: `reports/distill_evo_public_validation_latest.json`
5. 形式化闭环: `tools/run_distill_evolution_public_formal_assessment.py` 生成 `.lean` 与 assessment
6. 面板汇总: `tools/generate_one_click_kpi_dashboard.py` -> `reports/one_click_kpi_dashboard_latest.{json,md,png}`

## 5. 子系统构成（按功能视角）

### A. 编排与自治
- 代表文件: `evolution_system.py`, `agi_*_daemon.py`, `agi_*_evolution*.py`
- 作用: 管理演化循环、任务执行、状态持久化与策略迭代

### B. 核心数学/推理模块（h2q_core 族）
- 代表文件: `h2q_project/src/h2q/core/*`, `h2q_project/h2q/core/*`
- 主题: DDE、SST、群/流形结构、非交换几何/拓扑相关实验模块

### C. 服务与接口层
- 代表文件: `h2q_project/h2q_server.py`, `align_server.py`
- 作用: 将核心模块以 API 方式对外暴露，并接入守护/审计中间层

### D. 蒸馏与自评增强链路
- 代表文件:
  - `tools/collect_self_eval_distill_samples.py`
  - `tools/train_self_eval_distillation_adapter.py`
  - `tools/run_self_eval_distillation_pipeline.py`
  - `tools/run_self_model_consistency_benchmark.py`
  - `tools/trusted_local_agi_chat.py`
- 作用: 将失败样本 -> 教师样本 -> 适配器 -> 一致性提升闭环化

### E. 验证/门禁/发布
- 代表文件: `tools/release_gate.py`, `tools/public_alignment_report.py`, `tools/unified_system_framework.py`
- 作用: 统一门禁标准、对齐指标、可发布状态判断

### F. 监控与可观测性
- 代表文件: `tools/agi_realtime_monitor.py`, `tools/capability_registry.py`
- 作用: 周期快照、小时诊断、稳定性趋势追踪

### G. 形式化验证
- 代表文件:
  - `tools/run_distill_evolution_public_formal_assessment.py`
  - `reports/distill_evolution_logic_closure_latest.lean`
- 作用: 把关键结论转化为可编译逻辑闭包命题，作为“结构化证据层”

## 6. 报告工件体系（你需要重点关注的“真相层”）

### 6.1 latest 快照层（建议首看）
- `reports/*_latest.json`
- `reports/*_latest.md`

用途:
- 快速查看当前系统状态
- 避免被海量历史时间戳文件淹没

### 6.2 时间戳历史层
- 形如 `reports/<name>_<epoch>.json|md|png`
- 用途: 回溯实验轨迹、比较不同轮次演进

### 6.3 当前与你本轮工作最相关的关键工件
- `reports/self_eval_distillation_pipeline_latest.json`
- `reports/self_model_consistency_distilled_latest.json`
- `reports/distill_evo_public_validation_latest.json`
- `reports/distill_evo_public_formal_assessment_latest.json`
- `reports/one_click_kpi_dashboard_latest.md`
- `reports/research_aggregation_cross_validation_latest.md`
- `reports/research_aggregation_cross_validation_proof_note_latest.md`

## 7. 当前架构优势（工程视角）

1. 多证据链闭环已成型
- 蒸馏一致性、公开验证、门禁、形式化验证、KPI 汇总互相可引用。

2. 自动化脚本覆盖较全面
- 从采样到报告的“可执行路径”较完整，便于迭代与审计。

3. latest 机制利于运行态判断
- 对于超大工件仓库，`*_latest.*` 作为入口是高效设计。

4. 形式化验证已落地实践
- Lean4 逻辑闭包不是停留在文档层，已有脚本与工件串联。

## 8. 主要风险与管理挑战

1. 工件爆炸与认知负担
- `reports/` 文件量极高，导致日常定位成本大。

2. 入口与文档碎片化
- 根目录报告极多，存在“结论多源化”问题，易出现口径不一致。

3. 可选依赖/降级路径复杂
- Docker、外部 API、局部 fallback 共存，运行语义需要更强显式化。

4. 状态与任务堆积风险
- 持久状态随迭代增长，若缺少淘汰/归档策略，后续维护难度上升。

## 9. 建议的“整体把握”阅读/运维顺序

建议你固定按以下顺序看，能快速抓住全局:

1. `README.md`（Reality-First 边界）
2. `evolution_system.py`（主调度与状态机）
3. `h2q_project/h2q_server.py`（服务与核心模块连接）
4. `tools/release_gate.py`（门禁标准）
5. `tools/run_agi_integrated_validation.py`（综合验证）
6. `tools/run_distill_evolution_public_formal_assessment.py`（形式化层）
7. `tools/generate_one_click_kpi_dashboard.py`（统一展示层）
8. `reports/one_click_kpi_dashboard_latest.md`（当前全局快照）

## 10. 面向后续治理的三条优先动作

1. 建立工件分级与保留策略
- 保留 `latest + 关键里程碑`，其余按周期归档。

2. 建立“单一事实源”索引页
- 新建一个总索引（比如 `reports/SYSTEM_OVERVIEW_LATEST.md`）只链接核心 latest 工件。

3. 把关键论证纳入固定流水线
- 将“研究映射 + 交叉验证 + proof note”定时跑并纳入 release gate 证据集。

---

## 附录: 本次扫描方法说明
- 结构扫描: 顶层目录与子目录分类
- 入口扫描: 调度入口、服务入口、工具链入口
- 工件扫描: `reports/` latest 与时间戳工件
- 证据扫描: 蒸馏、验证、形式化、KPI、研究聚合报告

本报告用于“整体把握与项目治理”，不替代某个单模块的详细技术文档。
