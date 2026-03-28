# DAS 最终审计总报告（对外披露）

日期：2026-03-28
模型：distilgpt2
范围：Transformer 单头 Q/K/V 蒸馏 + token-table 蒸馏到 DAS 数学结构

## 一、披露目标

本轮披露聚焦三项可审计目标：

1. 将教师模型行为蒸馏到新数学结构并输出可加载权重文件。
2. 在压缩条件下保持推理加速。
3. 通过多随机种子、多秩 Pareto 审计给出置信区间与验收结论。

## 二、方法与新增工程变更

### 1) token 蒸馏目标强化

在 `h2q_project/tools/das_qkv_token_distill_experiment.py` 中引入以下增强项：

- 线性温度退火（temperature start -> end）
- top-k 排序损失（pairwise ranking hinge）
- KL + MSE + Ranking 的混合损失加权
- 可配置参数：`topk`, `ranking_weight`, `mse_weight`, `ranking_margin`

### 2) Pareto 审计参数透传

在 `h2q_project/tools/das_pareto_audit.py` 中增加超参透传，使多种子审计可复用强化蒸馏配置。

## 三、关键实验结果

## 3.1 单次定向冲刺（seed=42）

- rank32（强化配置 v2）：
  - top5 overlap = 0.5589
  - cosine = 0.99989
  - speedup = 3.1940x
  - compression = 17.4478x
- rank64（强化配置）：
  - top5 overlap = 0.5589
  - cosine = 0.99988
  - speedup = 2.7040x
  - compression = 8.7240x

说明：在定向配置下，rank32 与 rank64 均达到 top5 >= 0.55。

## 3.2 多种子 Pareto 审计（ranks=32,64; seeds=11,22,33）

强化配置：temp=1.35->0.12, topk=10, rank_w=0.75, mse_w=0.03, margin=0.08

- rank32：
  - mean top5 = 0.5325
  - 95% CI = [0.5156, 0.5495]
  - mean cosine = 0.999864
  - mean speedup = 1.7095x
  - mean compression = 17.4468x
  - consistency 验收：未通过（top5 阈值 0.55）
- rank64：
  - mean top5 = 0.6664
  - 95% CI = [0.6438, 0.6890]
  - mean cosine = 0.999919
  - mean speedup = 1.6021x
  - mean compression = 8.7238x
  - consistency 验收：通过

## 四、验收结论

阈值：cosine>=0.97, top5>=0.55, speedup>=1.05, compression>=2.0

- 全局结论：
  - 速度与压缩：rank32/rank64 全部通过。
  - 一致性：rank64 在多种子统计下通过；rank32 在多种子统计下尚未稳定通过。

工程结论：

1. 新数学结构权重蒸馏路线已成立（可训练、可导出、可加载、可加速、可审计）。
2. 若要求“多种子稳健通过 top5>=0.55”，当前推荐默认点为 rank64。
3. rank32 适合作为高压缩候选，但需进一步强化稳健性（可通过更长训练、难负样本策略或分层蒸馏改进）。

## 五、已达成项与未达成项（对外透明披露）

已达成：

- 完成公开单头 Q/K/V 蒸馏实验链路。
- 形成新结构权重文件与可读加载器链路。
- 给出多种子、多秩 Pareto 审计与 CI95 结果。
- 证明在严格压缩下可保持显著速度提升。

未完全达成：

- rank32 在多种子层面的 top5 稳定性尚未达 0.55 阈值。

原因分析：

- rank32 在压缩比更高时，token 排序细节易受种子扰动。
- 当前损失虽显著改善 top5，但对低秩容量边界仍存在波动。

## 六、可复验产物

- 审计 JSON：`reports/conv_math_conversion/das_pareto_audit_tuned/das_pareto_audit_20260328.json`
- 审计 Markdown：`reports/conv_math_conversion/das_pareto_audit_tuned/DAS_PARETO_AUDIT_20260328.md`
- 单次 rank32 冲刺：`reports/conv_math_conversion/rank32_tune_aggressive_v2/das_qkv_token_distill_distilgpt2_20260328.json`
- 单次 rank64 冲刺：`reports/conv_math_conversion/rank64_tune_aggressive/das_qkv_token_distill_distilgpt2_20260328.json`

---

披露声明：

本报告忠实披露“已通过项与未通过项”，未对未达标项进行删减或隐藏。审计口径可由同仓库脚本在相同参数下复现。
