# 同构转换可信工程审计结论（2026-03-28）

## A. 审计目标

验证以下命题是否成立：
1. 是否实现了权重架构同构变换且保持输出一致。
2. 在此基础上，是否实现稳定一致的压缩转换。
3. 在一致性前提下，是否出现可测得的推理加速。
4. 结论是否具备可复验、可审计、可验收属性。

## B. 审计证据

1. 严格同构实证
- `reports/conv_math_conversion/distilgpt2/conversion_report.json`
- `reports/conv_math_conversion/sshleifer__tiny-gpt2/conversion_report.json`

2. 加速与稳定性审计
- `reports/conv_math_conversion/acceleration_consistency_audit_20260328.json`

3. 算法与实现
- `h2q_project/tools/conv_math_weight_conversion_experiment.py`

## C. 审计门槛（Acceptance Gates）

1. Gate-G1（严格同构一致性）
- 要求：top1=1.0，top5=1.0（在审计样本上）。

2. Gate-G2（近似压缩稳定一致）
- 要求：top1>=0.95 且 top5>=0.90。

3. Gate-G3（运行时加速）
- 要求：平均延迟至少 1.05x 提升。

## D. 结果

1. G1：通过
- `permute_exact` 达到 token 级一致性 1.0。

2. G2：未通过
- `approx_conv_math` 在本次批次审计中 top1=0.0、top5=0.05，稳定性不足。

3. G3：未通过
- 速度比（相对原模型）：
  - `permute_exact = 0.9898x`
  - `approx_conv_math = 0.9077x`
- 结论：无可证加速，当前样本下反而更慢。

## E. 工程判定

1. 已完成能力
- 权重架构同构保持（函数不变）能力，已成立。

2. 未完成能力
- 无损压缩 + 稳定一致 + 推理加速，一体化目标尚未达成。

3. 可信声明边界
- 可声明："已实现并验证严格同构一致性"。
- 不可声明："已实现无损压缩并稳定加速"。

## F. 后续验收建议（必须条件）

1. 引入压缩态原生推理内核（低秩/分解态直接执行），避免重构回 dense。
2. 将近似路线一致性门槛固定为 top1/top5 与序列级指标联合约束。
3. 建立多批次随机种子审计，给出均值/方差与失败率。
4. 在同一硬件上执行端到端吞吐、延迟、内存三指标联合验收。
