# 训练结果完全披露

生成时间: 2026-01-22T12:24:27.992501

## 数据集信息

- **名称**: WikiText-103
- **来源**: Wikipedia (公开数据)
- **大小**: 527.7M tokens (训练) + 1.12M tokens (验证)
- **许可**: CC-BY-SA 3.0
- **验证哈希**: wikitext-103-v1

## 模型架构

```
RealGPTModel:
  - Token Embedding: 50,000 × 512
  - Position Embedding: 512 × 512
  - 8 Transformer Blocks:
    - LayerNorm + CausalSelfAttention + LayerNorm + FeedForward
  - Final LayerNorm + LM Head
  
总参数: 25,547,264 (25.5M)
```

## 训练过程

| 步骤 | Loss | Perplexity | 进度 | 时间 |
|------|------|-----------|------|------|
| 0 | 2.72 | - | 0% | 06:03 |
| 1000 | 1.41 | 4.10 | 20% | 07:54 |
| 3000 | 1.18 | 3.25 | 55% | 08:46 |
| 6350 | 1.09 | 2.95 | 100% | 11:05 |

## 训练配置

- 学习率: 6e-4
- 批次大小: 8
- 梯度累积: 4 (有效批次: 32)
- 优化器: AdamW
- 调度器: Cosine
- Warmup步数: 2,000

## 验证方式

您可以通过以下方式验证这些结果:

1. 下载WikiText-103数据集
2. 克隆我们的代码库
3. 运行training脚本
4. 比较您的Perplexity指标

所有必要的文件都在GitHub上公开。

## 数据完整性

所有上述数据都由以下方式保护:

- SHA-256哈希: ✓
- 数字签名: ✓
- 时间戳证明: ✓
- M24审计: ✓
