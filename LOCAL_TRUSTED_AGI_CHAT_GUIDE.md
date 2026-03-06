# 本地可运行 AGI 可信对话系统使用指南

## 1. 目标

本指南对应脚本 `tools/trusted_local_agi_chat.py`，实现以下流程：

1. 先做可信门控校验（真实调用 `tools/trusted_joint_agi_quantum_center.py`）。
2. 再连接本地大语言模型服务（`h2q_project/h2q_server.py` 的 `/chat`）。
3. 最后进入多轮对话，并保存会话与运行证据。

该流程的定位是“可复验、可追踪、可本地运行”的 AGI 对话系统基线。

## 2. 一键运行

在仓库根目录执行：

```bash
.venv/bin/python tools/trusted_local_agi_chat.py --force-refresh-trust
```

如果你已经有新鲜可信报告，可省略 `--force-refresh-trust`：

```bash
.venv/bin/python tools/trusted_local_agi_chat.py
```

## 3. 关键参数

- `--profile quick|full`：可信门控模式，`quick` 更快，`full` 更严格。
- `--skip-rsa`：跳过 RSA 并行交叉验证阶段。
- `--strict-trust-gate`：若信任分低于阈值则阻断对话。
- `--min-trust-score 0.70`：严格门控阈值。
- `--no-auto-start-server`：不自动拉起本地服务，要求你先手动启动。
- `--no-das-arch`：对话时不走 DAS 架构分支（默认走 DAS）。

## 4. 推荐真实通话流程

### 阶段 A：系统自检

1. 运行脚本，观察终端输出中的 `trust_score` 与 `trusted_ready`。
2. 确认 gates 至少包括：
- `das_decision_ready=True`
- `dual_aligned_consistent=True`
- `codec_integrity=True`

### 阶段 B：能力对话（基础）

建议先用 3 个任务确认可用性：

1. 解释类任务：
- 示例：`请解释 dual aligned 判据在本项目中的意义。`
2. 结构化任务：
- 示例：`请输出一个 JSON，包含目标、步骤、风险、验证项。`
3. 代码任务：
- 示例：`请给出一个最小 Python 函数，读取 reports 中最新 trusted center 报告并打印 trust_score。`

### 阶段 C：可信闭环

1. 每轮对话观察运行态字段：`status`、`fueter_curvature`、`spectral_shift_eta`。
2. 对关键回答执行复验（可让 AGI 给出可执行命令或脚本后直接本地运行）。
3. 会话结束后检查自动保存文件：
- `reports/trusted_local_agi_chat_session_<timestamp>.json`

## 5. 对话命令

交互时支持以下命令：

- `/help`：查看命令。
- `/status`：查看当前可信门控摘要。
- `/exit`：结束对话并保存会话记录。

## 6. 手动服务启动（可选）

如果你不想让脚本自动启动服务，可以先手动运行：

```bash
PYTHONPATH=. .venv/bin/python -m uvicorn h2q_project.h2q_server:app --host 127.0.0.1 --port 8000
```

然后新终端执行：

```bash
.venv/bin/python tools/trusted_local_agi_chat.py --no-auto-start-server
```

## 7. 产物与证据

运行完成后会生成以下证据：

1. 可信中心报告：`reports/trusted_joint_agi_quantum_center_*.json`
2. 对话会话记录：`reports/trusted_local_agi_chat_session_*.json`
3. 若自动拉起服务：`reports/trusted_local_agi_server_*.log`

这些文件共同构成“可信门控 + 本地推理 + 人机对话”的复验链路。
