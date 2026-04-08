# SmolLM2 -> DAS 新架构运行结构说明

日期：2026-03-28
模型：HuggingFaceTB/SmolLM2-360M

## 1. 目录结构（本次实验）

- reports/conv_math_conversion/sota_smallmodel_smollm2_360m/das_qkv_token_distill_HuggingFaceTB__SmolLM2-360M_20260328.json
- reports/conv_math_conversion/sota_smallmodel_smollm2_360m/das_token_structure_HuggingFaceTB__SmolLM2-360M_20260328.pt
- reports/conv_math_conversion/sota_smallmodel_smollm2_360m/das_token_structure_manifest_HuggingFaceTB__SmolLM2-360M_20260328.json
- reports/conv_math_conversion/sota_smallmodel_smollm2_360m/das_token_structure_loaded_summary_20260328.json
- reports/conv_math_conversion/sota_smallmodel_smollm2_360m/pareto_audit/das_pareto_audit_20260328.json

## 2. 权重文件与用途

1. `*.pt`：DAS 新数学架构权重包（QKV + token mapper）。
2. `*_manifest*.json`：结构配置与入口描述。
3. `*_distill*.json`：单次蒸馏训练与性能指标。
4. `pareto_audit/*.json`：多种子可信性验收指标。

## 3. 最小可运行加载命令

```bash
PYTHONPATH=. python3 h2q_project/tools/das_token_structure_loader.py \
  --structure reports/conv_math_conversion/sota_smallmodel_smollm2_360m/das_token_structure_HuggingFaceTB__SmolLM2-360M_20260328.pt \
  --output-json reports/conv_math_conversion/sota_smallmodel_smollm2_360m/das_token_structure_loaded_summary_20260328.json
```

## 4. 程序化调用（核心接口）

加载器实现：`h2q_project/tools/das_token_structure_loader.py`

- `load_das_token_structure(path, device)`：加载 DAS 权重包
- `DASTokenStructure.map_token_logits_subset(hidden)`：输出 token table 子空间 logits
- `DASTokenStructure.map_token_logits_full(hidden, vocab_size)`：回填到全词表 logits

## 5. 可用性验收结论（本模型）

- 结构可加载：通过
- 单次蒸馏指标：
  - token cosine = 0.96698
  - top5 overlap = 0.56696
  - speedup = 3.6772x
  - compression = 20.4181x
- 多种子审计（rank=32, seeds=11/22/33）：
  - mean cosine = 0.97005
  - mean top5 = 0.57321
  - mean speedup = 1.86966x
  - mean compression = 20.41705x
  - 一致性/速度/压缩：均通过
