# DAS 最终审计更新（rank32 稳健性冲刺）

日期：2026-03-28
关联基线报告：`reports/conv_math_conversion/DAS_FINAL_DISCLOSURE_AUDIT_20260328.md`

## 更新目标

针对 rank32 在多种子审计中未稳定达到 `top5>=0.55` 的问题，执行一轮稳健性冲刺：

- 引入 hard negative 挖掘
- 引入分段训练（stage 1 / stage 2）
- 保留温度退火与混合损失框架

## 关键实现

代码变更：

- `h2q_project/tools/das_qkv_token_distill_experiment.py`
  - 新增参数：`hard_neg_k`, `hard_neg_weight`, `stage_split`, `stage1_rank_scale`
  - 新增损失项：hard-negative ranking hinge
  - 新增策略：分段训练下 ranking 权重与 margin 动态调度
- `h2q_project/tools/das_pareto_audit.py`
  - 支持上述新参数透传与报告记录

## 审计结果（rank32, seeds=11/22/33）

来源：

- `reports/conv_math_conversion/das_pareto_audit_rank32_hardneg/das_pareto_audit_20260328.json`
- `reports/conv_math_conversion/das_pareto_audit_rank32_hardneg/DAS_PARETO_AUDIT_20260328.md`

结果：

- mean cosine = 0.99967
- mean top5 = 0.68683
- top5 95% CI = [0.67041, 0.70324]
- mean speedup = 1.72258x
- mean compression = 17.44680x

阈值判定（cosine>=0.97, top5>=0.55, speedup>=1.05, compression>=2.0）：

- consistency: 通过
- speedup: 通过
- compression: 通过

## 结论更新

rank32 在“hard negative + 分段训练”配置下已实现多种子稳健通过，原先 rank32 未达标问题已关闭。
