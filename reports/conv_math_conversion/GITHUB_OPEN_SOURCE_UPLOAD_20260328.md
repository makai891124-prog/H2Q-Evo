# GitHub 开源上传说明（DAS 蒸馏权重）

日期：2026-03-28
分支：copilot/vscode-mmdt291x-9e7t

## 1. 已公开上传的核心文件

发布包目录：

- releases/das-smollm2-360m-das-v1/README.md
- releases/das-smollm2-360m-das-v1/das_token_structure_HuggingFaceTB__SmolLM2-360M_20260328.pt
- releases/das-smollm2-360m-das-v1/das_token_structure_manifest_HuggingFaceTB__SmolLM2-360M_20260328.json
- releases/das-smollm2-360m-das-v1/loader.py
- releases/das-smollm2-360m-das-v1/example_infer.py
- releases/das-smollm2-360m-das-v1/metrics_single_run.json
- releases/das-smollm2-360m-das-v1/metrics_multiseed.json
- releases/das-smollm2-360m-das-v1/requirements.txt
- releases/das-smollm2-360m-das-v1/publish_to_huggingface.sh

实验与审计目录：

- reports/conv_math_conversion/sota_smallmodel_smollm2_360m/das_qkv_token_distill_HuggingFaceTB__SmolLM2-360M_20260328.json
- reports/conv_math_conversion/sota_smallmodel_smollm2_360m/das_token_structure_HuggingFaceTB__SmolLM2-360M_20260328.pt
- reports/conv_math_conversion/sota_smallmodel_smollm2_360m/das_token_structure_manifest_HuggingFaceTB__SmolLM2-360M_20260328.json
- reports/conv_math_conversion/sota_smallmodel_smollm2_360m/das_token_structure_loaded_summary_20260328.json
- reports/conv_math_conversion/sota_smallmodel_smollm2_360m/pareto_audit/das_pareto_audit_20260328.json
- reports/conv_math_conversion/sota_smallmodel_smollm2_360m/pareto_audit/DAS_PARETO_AUDIT_20260328.md
- reports/conv_math_conversion/sota_smallmodel_smollm2_360m/DAS_SOTASMALL_RUNTIME_STRUCTURE_20260328.md

## 2. 文件用途说明

1. `*.pt`：新数学架构（DAS）蒸馏权重包。
2. `*manifest*.json`：结构元数据与运行配置。
3. `loader.py`：加载与推理接口。
4. `example_infer.py`：最小运行示例。
5. `metrics_single_run.json`：单次训练验收结果。
6. `metrics_multiseed.json`：多种子可信性结果。

## 3. 计算优势（当前公开口径）

SmolLM2-360M -> DAS（rank=32）多种子审计结果：

- mean cosine: 0.97005
- mean top5 overlap: 0.57321
- mean speedup ratio: 1.86966x
- mean compression ratio: 20.41705x

阈值判定：一致性/速度/压缩均通过。

## 4. 下载后最小运行步骤

```bash
cd releases/das-smollm2-360m-das-v1
pip install -r requirements.txt
python example_infer.py
```

## 5. 边界说明

- 本权重为 DAS 蒸馏结构权重，不是原始教师模型全参数检查点。
- 对外宣称以本仓库可复验指标为准，不外推为硬件级量子优势复现。
