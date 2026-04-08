# 开源模型社区发布守则（DAS 蒸馏权重）

日期：2026-03-28
适用社区：Hugging Face Hub / ModelScope / GitHub Releases（优先 Hugging Face）

## 1. 必须满足的合规项

1. 许可证兼容性：
- 明确基础教师模型许可证（例如 SmolLM2 的许可证）
- 明确蒸馏权重发布许可证
- 若基础许可证要求保留声明或 NOTICE，必须完整附带

2. 来源可追溯：
- 记录教师模型 ID 与版本（commit/revision）
- 记录训练脚本版本与关键超参
- 记录数据来源与使用边界（公开语料、合成数据、私有数据是否涉及）

3. 安全与责任披露：
- 模型卡中写明用途、限制、已知失效场景
- 不夸大为“同规模硬件优势复现”
- 提供误用风险提示（生成错误信息、偏见、幻觉等）

## 2. 让他人“直接下载可用”的最小发布包

建议仓库根目录包含：

1. `README.md`（Model Card）
2. `LICENSE`
3. `NOTICE`（如上游许可证要求）
4. `das_token_structure_*.pt`（或 safetensors）
5. `das_token_structure_manifest_*.json`
6. `loader.py`（或引用本仓库 loader 路径）
7. `requirements.txt`（最小依赖版本）
8. `example_infer.py`（5~20 行最小推理样例）
9. `metrics.json`（单次+多种子验收）

## 3. Model Card 最低必填内容

1. 模型概述：
- 这是“DAS 新数学架构蒸馏权重”，不是原始 Transformer 全参数权重

2. 训练与蒸馏配置：
- teacher model
- 蒸馏损失组成（KL/MSE/ranking/hard-negative）
- 关键超参（温度退火、topk、rank 等）

3. 评测结果：
- 单次结果 + 多种子 CI
- 验收阈值与是否通过

4. 适用范围与边界：
- 适用任务
- 不适用任务
- 失败样本（例如极小模型不一定受益）

## 4. 文件格式建议

1. 优先 `safetensors`：
- 降低反序列化安全风险
- 若暂时使用 `.pt`，应在文档中明确 `torch.load` 安全注意事项

2. 保持 manifest 稳定：
- `format_version`
- `qkv_config`
- `token_mapper_config`
- `token_ids`

3. 版本语义化：
- 建议 tag：`vX.Y.Z-das-<teacher>-<date>`

## 5. Hugging Face 发布流程（推荐）

1. 新建模型仓库（public）
2. 上传上述最小发布包
3. 在 README 写清：
- 一行安装命令
- 一条加载命令
- 一段最小推理示例
4. 打上标签：
- `library_name: pytorch`
- `pipeline_tag`（按任务）
- `license`
5. 启用讨论区并固定“已知问题与路线图”

## 6. 可信发布的工程建议

1. 同时发布通过样本与失败样本（透明边界）
2. 发布可复现实验命令（含 seeds）
3. 固定依赖版本（避免环境漂移）
4. 提供 SHA256 校验（权重与清单）

## 7. 对本项目当前可直接使用的结论

针对 `HuggingFaceTB/SmolLM2-360M -> DAS`：

- 已有可用权重文件与 manifest
- 已有加载器与加载成功摘要
- 已有多种子验收报告（通过）

因此已具备“可在开源社区直接下载并运行”的技术基础；剩余工作主要是发布合规与模型卡完善。
