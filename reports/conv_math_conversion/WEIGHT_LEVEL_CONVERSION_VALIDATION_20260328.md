# 开源小模型权重级转换验证报告（2026-03-28，更新）

## 1. 目标

按“下载开源模型 -> 数学映射转换 -> 中间件对齐 -> 一致性与可用性验证”的链路，给出可复验、可审计的权重级转换结论，并明确哪些结论是严格成立、哪些仍是近似工程路线。

## 2. 当前已实现的两类算法

1. 近似压缩转换（`approx`）
- 卷积平滑 + 低秩分解 + 四元数流形归一化 + 残差低秩校正 + KL 校正蒸馏。
- 目标：压缩与迁移。
- 特征：非严格同构，输出一致性需靠校正训练逼近。

2. 严格同构转换（`permute_exact`）
- 隐藏维自反置换同构（involution permutation）。
- 目标：函数级完全等价。
- 特征：不追求压缩，但可给出数学上严格的一致性保证。

## 3. 严格同构算法（真实可靠）

设隐藏状态按行向量记为 $h \in \mathbb{R}^{d}$，取置换矩阵 $P$，且满足 $P^{-1}=P$（自反置换）。定义新坐标

$$
h' = hP
$$

对 GPT2 类模块做如下参数变换（Conv1D 记为 $y=xW+b$）：

1. hidden -> hidden:
$$
W' = P^{-1}WP,\quad b' = bP
$$

2. hidden -> m:
$$
W' = P^{-1}W,\quad b' = b
$$

3. m -> hidden:
$$
W' = WP,\quad b' = bP
$$

4. LayerNorm:
$$
\gamma' = \gamma P,\quad \beta' = \beta P
$$

5. Embedding / Positional Embedding:
$$
E' = EP
$$

6. LM Head（线性层权重）:
$$
W_{lm}' = W_{lm}P^{-1}
$$

当输入 embedding 与输出 head 绑权重（tied weights）时，要求 $P=P^{-1}$，因此采用自反置换是必要条件。

结论：按上述规则全链路变换后，网络表示函数保持不变（仅坐标系重标号），因此 logits 与生成决策严格一致。

## 4. 最新实测结果

来源：
- `reports/conv_math_conversion/sshleifer__tiny-gpt2/conversion_report.json`
- `reports/conv_math_conversion/distilgpt2/conversion_report.json`

`permute_exact` 模式结果：

1. `sshleifer/tiny-gpt2`
- `avg_cosine_last_logits = 1.0000`
- `avg_top1_match = 1.0000`
- `avg_top5_overlap = 1.0000`

2. `distilgpt2`
- `avg_cosine_last_logits = 1.0000`
- `avg_top1_match = 1.0000`
- `avg_top5_overlap = 1.0000`

这给出“可复验的严格同构一致性”证据：在开放权重小模型上，权重级映射可实现严格等价。

## 5. 与近似压缩路线的关系

1. 严格同构路线（已证实）
- 成功目标：保持功能完全不变。
- 代价：本身不压缩参数规模。

2. 近似压缩路线（仍在优化）
- 成功目标：降低内存/计算，同时尽量保持行为。
- 当前状态：通过校正蒸馏显著提升一致性，但尚未达到“严格等价”。

因此，当前可给出的“真实可靠”结论是：
- 严格同构可行（已实证）。
- 压缩且严格等价，尚未被当前近似链路证明。

## 6. 边界与工程真相

1. `permute_exact` 是“等价重参数化”，不是压缩。
2. 若目标是降显存/降算力，仍需在严格同构基线上叠加可控近似（低秩/量化/蒸馏）。
3. 高余弦不代表生成一致，必须联合 top-1/top-k 与序列级指标。
4. tied embedding/head 会约束可用变换群，任意可逆变换并不总可行。

## 7. 研究脉络（用于方法定位）

1. 函数保持变换：Net2Net（arXiv:1511.05641）。
2. 蒸馏校正：Distilling the Knowledge in a Neural Network（arXiv:1503.02531）。
3. 低秩适配：LoRA（arXiv:2106.09685）。
4. 张量分解加速：CP-decomposition + fine-tuning（arXiv:1412.6553）。

## 8. 产物索引

- 实验脚本：`h2q_project/tools/conv_math_weight_conversion_experiment.py`
- tiny-gpt2 报告：`reports/conv_math_conversion/sshleifer__tiny-gpt2/conversion_report.json`
- distilgpt2 报告：`reports/conv_math_conversion/distilgpt2/conversion_report.json`
- 转换模型目录（示例）：`reports/conv_math_conversion/distilgpt2/converted_model/`

## 9. 加速与稳定性审计（新增）

来源：`reports/conv_math_conversion/acceleration_consistency_audit_20260328.json`

审计设置：
- 模型：`distilgpt2`
- 设备：CPU
- 批量：8，序列长度：13
- 比较对象：原模型 / `permute_exact` / `approx`

关键结果：

1. 严格同构（`permute_exact`）
- 一致性：
	- `avg_cosine_last_logits = 0.99999994`
	- `avg_top1_match = 1.0`
	- `avg_top5_overlap = 1.0`
- 速度比（相对原模型）：`0.9898x`（未加速，约等速略慢）

2. 近似压缩（`approx_conv_math`）
- 一致性：
	- `avg_cosine_last_logits = 0.9954`
	- `avg_top1_match = 0.0`
	- `avg_top5_overlap = 0.05`
- 速度比（相对原模型）：`0.9077x`（无加速，反而更慢）

审计判定：
- `exact_isomorphism_lossless = true`
- `approx_stably_equivalent = false`
- `runtime_acceleration_observed = false`

## 10. 工程可验收结论（当前版本）

1. 已被证实：
- 我们已经实现“改变权重坐标架构但函数保持不变”的严格同构方案（`permute_exact`）。
- 在开放小模型上可复验达到 token 级 1.0 一致性。

2. 尚未被证实：
- “无损压缩且稳定一致且推理加速”这一更强命题，在当前实现中不成立。

3. 根因：
- 当前 `approx` 路线在推理时仍回到 dense 权重执行，尚无压缩态原生 kernel。
- 因而即便有压缩表示，也不会自动带来运行时加速。

4. 可信性分析结论：
- 严格同构能力：通过。
- 稳定近似等价能力：未通过。
- 运行时加速能力：未通过。

这意味着：当前阶段可宣称“同构保持能力已建立”，但不能宣称“无损压缩加速已完成”。

