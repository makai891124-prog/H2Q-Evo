# H2Q-Evo 架构速览 Mindmap（管理层 10 分钟版）

- 版本时间: `2026-03-07`
- 目标: 用最短时间看清“系统是什么、怎么跑、风险在哪、接下来做什么”

## 1) 战略定位
- 项目类型: 自演化实验系统 + 验证工厂 + 报告工件平台
- 核心价值: 把“改进动作”转化为“可验证证据链”
- 当前成熟度: 工程链路完整，治理与收敛策略待强化

## 2) 三层结构总图

### A. 代码层（Build）
- 核心实现: `h2q_project/`
- 编排入口: `evolution_system.py`
- 工具自动化: `tools/`

### B. 运行层（Run）
- 推理服务: `h2q_project/h2q_server.py`
- 演化循环: `evolution_system.py` + `agi_*daemon.py`
- 门禁/验证: `tools/release_gate.py`, `tools/run_agi_integrated_validation.py`

### C. 证据层（Prove）
- latest 快照: `reports/*_latest.json|md`
- KPI 汇总: `reports/one_click_kpi_dashboard_latest.md`
- 形式化闭环: `reports/distill_evolution_logic_closure_latest.lean`
- 学术化交叉验证: `reports/research_aggregation_cross_validation_latest.md`

## 3) 关键业务流程（从动作到证据）
1. 采样与训练
- `tools/collect_self_eval_distill_samples.py`
- `tools/train_self_eval_distillation_adapter.py`

2. 一致性与稳健性验证
- `tools/run_self_model_consistency_benchmark.py`
- 关键输出: `reports/self_model_consistency_distilled_latest.json`

3. 集成与公开验证
- `tools/run_agi_integrated_validation.py`
- 关键输出: `reports/distill_evo_public_validation_latest.json`

4. 形式化证明
- `tools/run_distill_evolution_public_formal_assessment.py`
- 关键输出: `reports/distill_evo_public_formal_assessment_latest.json`

5. 管理视图
- `tools/generate_one_click_kpi_dashboard.py`
- 关键输出: `reports/one_click_kpi_dashboard_latest.md`

## 4) 当前健康度快照（管理关注）
- 仓库规模较大: 多层历史工件 + 多入口脚本
- 自动化覆盖较高: 采样、训练、验证、门禁、可视化、形式化基本贯通
- 风险主要不在“能不能跑”，而在“长期治理成本”

## 5) 四大管理风险

### 风险 1: 工件过载
- 现象: `reports/` 体量很大
- 影响: 查找成本高，认知噪声大
- 管理信号: 需要明确保留/归档策略

### 风险 2: 文档碎片化
- 现象: 根目录历史总结和报告数量多
- 影响: 决策口径可能不一致
- 管理信号: 需要“单一事实源”

### 风险 3: 路径复杂性
- 现象: API、本地、Docker、fallback 并存
- 影响: 故障定位与复现成本上升
- 管理信号: 需要模式化开关与标准运行剖面

### 风险 4: 状态积累
- 现象: 持久状态与任务历史长期增长
- 影响: 性能和维护风险递增
- 管理信号: 需要生命周期治理

## 6) 管理层建议（优先级）

### P0（先做）
- 建立工件生命周期策略（latest 常驻 + 历史归档）
- 建立统一状态与任务治理（清理、重试、死亡队列）
- 建立单一索引页（统一引用核心 latest 证据）

### P1（再做）
- 收敛运行模式（标准化 profile，明确 fallback）
- 合并重复验证链，减少重复计算
- 加强关键路径测试覆盖与故障演练

### P2（持续）
- 指标平台化（趋势、告警、回归）
- 模型与实验产物注册化（版本、血缘、清理策略）

## 7) 管理层每周最小检查清单
1. 门禁状态是否持续为 True
- `reports/release_gate_latest.md`

2. 公开验证是否稳定
- `reports/distill_evo_public_validation_latest.json`

3. 形式化闭环是否通过
- `reports/distill_evo_public_formal_assessment_latest.md`

4. KPI 是否出现回退
- `reports/one_click_kpi_dashboard_latest.md`

5. 交叉验证结论是否保持 robust
- `reports/research_aggregation_cross_validation_latest.md`

## 8) 一句话结论
H2Q-Evo 已具备“工程执行 + 证据验证”的闭环能力，下一阶段成功关键在于“治理收敛”，而不是继续无约束扩展脚本与工件。
