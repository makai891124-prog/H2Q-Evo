# H2Q-Evo 重构优先级 Backlog（P0/P1/P2）

- 版本时间: `2026-03-07`
- 目标: 给出可执行、可验收、可排期的重构清单
- 范围: 以当前主链路（演化 -> 验证 -> 形式化 -> KPI）为核心

## 执行原则
- 先治理复杂度，再扩展能力
- 先稳定证据链，再追求新指标
- 每项任务必须有 DoD（Definition of Done）

## P0（0-2 周）：立刻降低系统复杂度与运营风险

### P0-1 工件生命周期治理
- 目标: 解决 `reports/` 过载，保留决策必需证据
- 动作:
- 制定工件分级: `latest`, `milestone`, `archive`
- 自动归档时间戳工件到 `reports/archive/YYYY-MM/`
- 为 latest 工件建立白名单（仅核心 20-30 个）
- 交付物:
- `tools/report_retention_manager.py`
- `reports/REPORT_RETENTION_POLICY_latest.md`
- DoD:
- 每日自动执行后，`reports/` 增量可控
- 能在 2 分钟内定位全部核心 latest 证据

### P0-2 单一事实源索引页
- 目标: 减少文档口径分散
- 动作:
- 新建统一索引 `reports/SYSTEM_OVERVIEW_latest.md`
- 固定引用核心 latest 工件，不直接引用历史时间戳
- 交付物:
- `reports/SYSTEM_OVERVIEW_latest.md`
- DoD:
- 管理层仅需阅读该页即可获取当前状态
- 与 one-click KPI、formal assessment、public validation 链接一致

### P0-3 状态与任务治理
- 目标: 避免状态无限膨胀和失败任务堆积
- 动作:
- 为任务状态增加 `retry_count`、`last_attempt_at`、`dead_letter`
- 增加失败任务降噪策略（退避重试 + 最大重试）
- 交付物:
- `evolution_system.py` 状态机增强
- `reports/TASK_GOVERNANCE_STATUS_latest.md`
- DoD:
- 连续运行 72 小时无明显失败任务堆积
- dead letter 队列可追溯且可重放

### P0-4 运行模式标准化
- 目标: 收敛 API/Local/Docker/fallback 的行为差异
- 动作:
- 定义标准 profile: `dev`, `quick-validate`, `release`
- 输出每次运行的模式摘要到报告
- 交付物:
- `tools/runtime_profile_manager.py`
- `reports/RUNTIME_PROFILE_latest.json`
- DoD:
- 同一 profile 多次运行结果可复现
- 故障日志明确标识当前模式和降级路径

## P1（2-6 周）：强化可维护性与验证效率

### P1-1 验证流水线去重
- 目标: 降低重复计算和脚本重叠
- 动作:
- 合并重复验证步骤（release gate / integrated validation / one-click）
- 统一中间产物 schema
- 交付物:
- `tools/run_validation_pipeline.py`（统一入口）
- `reports/VALIDATION_SCHEMA_SPEC_latest.md`
- DoD:
- 验证总时长下降 >= 30%
- 同一轮次不再重复生成同类统计

### P1-2 关键链路测试补齐
- 目标: 给核心主链路加防回归护栏
- 动作:
- 为以下脚本补充聚焦测试:
- `tools/run_self_eval_distillation_pipeline.py`
- `tools/run_agi_integrated_validation.py`
- `tools/run_distill_evolution_public_formal_assessment.py`
- 交付物:
- `tests/test_distill_pipeline.py`
- `tests/test_integrated_validation.py`
- `tests/test_formal_assessment.py`
- DoD:
- CI 覆盖主链路关键断点
- 新增变更引发回归时可在 CI 中被拦截

### P1-3 指标口径统一
- 目标: 统一 KPI、验证、形式化报告中的同名指标定义
- 动作:
- 建立指标字典与计算来源表
- 自动校验跨报告一致性
- 交付物:
- `reports/METRIC_DICTIONARY_latest.md`
- `tools/validate_metric_consistency.py`
- DoD:
- 同名指标在多报告中的数值/解释一致

## P2（6-12 周）：平台化与长期演进能力

### P2-1 证据链平台化
- 目标: 从“文件工件”升级到“可查询证据图谱”
- 动作:
- 建立轻量索引数据库（SQLite）记录每次运行的证据关系
- 支持从 KPI 反向追踪到原始工件
- 交付物:
- `tools/build_evidence_index.py`
- `reports/EVIDENCE_INDEX_latest.db`
- DoD:
- 任一结论可在 1 分钟内反查原始证据路径

### P2-2 模型与实验资产注册
- 目标: 管理模型权重与实验血缘
- 动作:
- 统一记录权重版本、训练来源、评测结果
- 清理失效检查点，保留关键里程碑
- 交付物:
- `tools/model_registry.py`
- `reports/MODEL_REGISTRY_latest.json`
- DoD:
- 任一线上权重可追溯其训练与验证来源

### P2-3 形式化验证扩展
- 目标: 从单一闭包命题扩展到可组合规则集
- 动作:
- 将 Lean 命题模块化（门禁、稳定性、鲁棒性）
- 增加失败时降级证明策略
- 交付物:
- `reports/formal_rules/`
- `tools/run_formal_suite.py`
- DoD:
- 形式化验证失败时可定位到规则级别

## 建议排期（可直接执行）
- 第 1 周: P0-1, P0-2
- 第 2 周: P0-3, P0-4
- 第 3-4 周: P1-1
- 第 5 周: P1-2
- 第 6 周: P1-3
- 第 7-10 周: P2-1, P2-2
- 第 11-12 周: P2-3

## 每周验收模板（执行层）
1. 本周完成项
- 列出完成 backlog 编号与交付物路径

2. 本周风险
- 列出阻塞项与缓解动作

3. 下周计划
- 列出具体编号与验收标准

4. 关键指标
- 运行时长
- 失败率
- latest 工件数量
- 关键门禁通过率

## 一句话执行建议
先把“证据链治理”做稳，再扩展新能力模块；否则系统复杂度会先于能力增长失控。
