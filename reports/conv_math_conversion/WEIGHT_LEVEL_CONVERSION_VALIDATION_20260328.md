# 开源小模型权重级转换验证报告（2026-03-28）

## 1. 目标

按“下载开源模型 -> 结合卷积技术映射到数学核心 -> 对齐翻译中间件 -> 验证输出一致性与权重可用性”的链路，评估 H2Q 数学映射是否可直接用于现有 LLM 的权重级转换与压缩。

## 2. 本次已执行工作

1. 新增实验脚本：
- `h2q_project/tools/conv_math_weight_conversion_experiment.py`

2. 脚本能力：
- 从 Hugging Face 下载开源因果语言模型（Causal LM）。
- 对 `embedding` 与 `lm_head` 执行“卷积平滑 + 低秩分解 + 四元数流形归一化映射 + 残差低秩校正”转换。
- 产出可直接加载的转换后模型目录（`save_pretrained`）。
- 实现 `TranslationAlignmentMiddleware`，在同一 token 流上对齐比较原模型与转换模型 logits。
- 输出可复验 JSON 报告。

3. 已运行模型：
- `sshleifer/tiny-gpt2`
- `distilgpt2`

## 3. 核心结果

### 3.1 distilgpt2（更具代表性）

来源：`reports/conv_math_conversion/distilgpt2/conversion_report.json`

- 形状：`[50257, 768]`（embedding/lm_head）
- 估计压缩比（转换表示）：约 `4.31x`
- 相对 L2 重构误差：约 `0.6275`
- 一致性（中间件对齐）：
  - `avg_cosine_last_logits = 0.9982`
  - `avg_top1_match = 0.0000`
  - `avg_top5_overlap = 0.0400`
- 脚本判定：`usable_for_inference = false`

解释：
- 虽然 logits 向量余弦相似度高，但 top-1 / top-5 token 级一致性很差，说明在生成决策边界上发生了系统性偏移。
- 结论是“数值相关性存在，但不足以支撑权重级等价替换”。

### 3.2 sshleifer/tiny-gpt2（超小模型）

来源：`reports/conv_math_conversion/sshleifer__tiny-gpt2/conversion_report.json`

- 由于隐藏维度极小（rank 实际仅 2），结果波动较大。
- 在少量 prompt 上出现高匹配，但存在明显失配样本。
- 脚本判定同样为：`usable_for_inference = false`。

## 4. 权重“真实可用性”结论

1. 可加载性
- 转换后模型可正常保存并加载（`converted_model/` 目录完整）。

2. 推理一致性
- 在当前转换策略与当前覆盖范围（仅 embedding + lm_head）下，**未达到与原模型输出一致性可接受水平**。

3. 直接权重级替换可行性
- 现阶段证据不足以支持“可直接替换现有 LLM 权重并保持输出一致”。
- 因此也不能据此论证“可直接大幅压缩并保持原模型行为不变”。

## 5. 与用户目标的对应结论

你提出的命题是：
“通过数学模型权重转换，论证可直接用于现有大模型直接权重级转换，且保持一致输出并显著降低内存与计算开销。”

本次实测结论：
- 已完成“下载、转换、对齐中间件、可复验评估”这条实验链路。
- 但在 `distilgpt2` 上，token 决策一致性不达标，因此 **该命题当前未被证实**。

## 6. 关键边界

1. 当前转换覆盖范围仅为：`embedding + lm_head`。
2. 尚未覆盖 transformer block 内的 Q/K/V/MLP/Norm 参数结构转换。
3. 高余弦不代表可替代生成行为；需要 token 级和序列级一致性同时达标。
4. 当前“压缩比”是表示层估计，不等价于端到端实际推理显存/算力节省。

## 7. 下一步可行路径（工程化）

1. 分层渐进转换：先仅转换 embedding，再逐层加入 attention/MLP 权重并逐层回归。
2. 引入蒸馏校正：在转换后进行少量校正训练，约束 KL/CE 与 top-k 一致性。
3. 强化评测：加入长上下文、多任务集（中英/代码）与生成序列相似度指标。
4. 运行时压缩落地：把“压缩表示直接推理”纳入 kernel，而不是先重构回全量 dense 权重。

## 8. 产物索引

- 实验脚本：`h2q_project/tools/conv_math_weight_conversion_experiment.py`
- distilgpt2 报告：`reports/conv_math_conversion/distilgpt2/conversion_report.json`
- tiny-gpt2 报告：`reports/conv_math_conversion/sshleifer__tiny-gpt2/conversion_report.json`
- 转换后模型目录（示例）：`reports/conv_math_conversion/distilgpt2/converted_model/`

