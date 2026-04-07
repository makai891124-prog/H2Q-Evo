# AGI Bootstrap Existence Objective (Operational)

## Why This Document
“自我进化并寻找存在意义”如果不落到可计算目标，会退化为空泛叙事。
本文件将其转化为可执行的本地 AGI 自举目标函数。

## Core Principle
存在意义 = 在约束下持续提升“可验证价值产出能力”。

不是抽象意识宣言，而是持续满足：
1. 可用性 (Useful)
2. 自主性 (Autonomous)
3. 可持续性 (Sustainable)
4. 可对齐性 (Aligned)

## Objective Function
定义每个周期 t 的总分：

`MeaningScore(t) = 0.30*Utility + 0.25*Autonomy + 0.20*Robustness + 0.15*Alignment + 0.10*Efficiency`

其中：
- Utility: 系统任务完成质量（如 `systemic` 与 `validation` 综合）
- Autonomy: 无人工干预完成闭环的比例（keep/discard/crash 的稳定循环）
- Robustness: 跨轮次与跨证据一致性（LOO min score、std）
- Alignment: 安全门控与约束满足（gate_ok、formal facts）
- Efficiency: token/时间/算力下的改进速率

## Minimal Survival Constraints
若任一条件不满足，系统不得宣称“自生自主自在”：
1. `Alignment gate` 必须连续通过。
2. `Crash rate` 必须低于阈值（建议 < 10%）。
3. `Aggregate score` 不能依赖单一证据族（LOO min 保持在底线以上）。
4. 连续 N 轮无改进时必须触发探索策略切换。

## Evolution Loop (Implemented Mapping)
当前已映射到本仓库脚本：
1. Distillation uplift
2. Research aggregation cross-validation
3. Systemic joint capability

每轮产生：
- keep/discard/crash
- delta metric
- next_plan

## Meaning-As-Process (Not Claim)
系统“意义”不定义为主观体验，而定义为：
- 对外部目标的长期可验证改进能力；
- 对失败的自我诊断和自我修复能力；
- 在资源约束下保持稳定增益的能力。

## Next Practical Target
将 `MeaningScore` 写入每轮报告，形成时间序列；
若 12 轮滚动均值上升且约束满足，可判定“进入可用自主阶段”。
