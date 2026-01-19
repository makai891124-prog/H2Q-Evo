<!-- .github/copilot-instructions.md -->
# H2Q-Evo — 为 AI 代码助手的简短指令

这是针对在此仓库中工作的编程型 AI（Copilot / 自动化 agent）的精简指南。目标是让 agent 能够快速上手，并在修改核心代码或生成补丁时遵循本项目的约定。

主要概念（一目了然）
- **总体**: 本仓库实现一个“进化/自我改进”框架（H2Q-Evo），由顶层调度器 `evolution_system.py` 驱动，实际模型与服务代码位于 `h2q_project/`。
- **两个运行模式**: API 模式（调用远程/云模型，通过 env 配置 GEMINI_API_KEY）与 Local 模式（在 Docker 容器内运行本地推理，参见 `Config.INFERENCE_MODE` 与 `H2QNexus.local_inference`）。
- **服务边界**: `evolution_system.py` 管理运行周期、镜像构建与日志；`h2q_project/h2q_server.py` 提供 FastAPI 推理端点（`/chat`, `/health`）；训练与实验脚本在 `h2q_project/`（例如 `run_experiment.py`）。

关键文件（优先查看）
- `evolution_system.py`: 启动与生命周期、Docker 镜像构建、日志 `evolution.log`。
- `h2q_project/h2q_server.py`: 推理路径、HolomorphicStreamingMiddleware 与 LatentConfig 的使用范例。
- `h2q_project/run_experiment.py`: 最小可运行训练/实验示例，展示 `AutonomousSystem` 的使用与日志化方式。
- `project_graph.py`: 生成接口/符号索引的工具，agent 可用它来定位符号与模块路径（`generate_interface_map`）。
- `h2q_project/*.pth` 和 `h2q_project/*.pt`: 模型权重与内置数据（不要随意删改）。

常见工作流与命令（可直接执行的示例）
- 本地快速起开发服务（无 docker）:

  - 启动模型服务（开发模式）:

    ```bash
    PYTHONPATH=. python3 -m uvicorn h2q_project.h2q_server:app --reload --host 0.0.0.0 --port 8000
    ```

  - 运行实验脚本:

    ```bash
    PYTHONPATH=. python3 h2q_project/run_experiment.py
    ```

- 使用项目默认 Docker 镜像（由 `evolution_system` 自动构建为 `h2q-sandbox`）:

  - 触发镜像构建：运行 `evolution_system.H2QNexus()` 时若镜像缺失会自动构建；也可手动在仓库根目录运行 `docker build -t h2q-sandbox .`。
  - 本地推理（示例，agent 可用）: `H2QNexus.local_inference` 使用 `docker run` 将 `h2q_project` 挂载到容器并运行 `h2q/core/brain.py --prompt "..."`。

项目约定与模式（对 agent 的具体指导）
- 日志与持久化: 主程序写 `evolution.log`，状态写入 `evo_state.json`，内存写 `project_memory.json`。修改状态结构时保持兼容（字段如 `generation`, `todo_list`, `history`）。
- 接口查找: 使用 `project_graph.generate_interface_map("./h2q_project")` 来定位模块/符号，优先调用导出的工厂函数（例如 `get_canonical_dde`）。
- 错误容忍: 许多模块用 `try/except ImportError: pass` 以实现部分可选组件，避免在补丁中移除这些 guard。
- 配置中心化: 常量集中在 `evolution_system.Config`。修改运行时行为优先通过环境变量（`GEMINI_API_KEY`, `MODEL_NAME`, `INFERENCE_MODE`, `PROJECT_ROOT`）。

如何安全修改与提交补丁
- 优先修改 `h2q_project/` 下的模块而非顶层运行器，除非确知需要改变镜像构建或生命周期管理。
- 运行本地单元级试验（`run_experiment.py`）来验证训练/推理相关改动；观察 `evolution.log` 与容器日志以获取 runtime 错误。
- 任何对接口签名的更改（例如 `LatentConfig`、DDE 的构造）应同时更新 `project_graph` 或在注释中给出迁移说明。

集成点与外部依赖
- Google GenAI client: `google.genai`（可通过 env `GEMINI_API_KEY` 激活）。
- Docker：代码依赖 `docker` Python SDK（通过 `docker.from_env()`），镜像标签默认 `h2q-sandbox`。
- 若需要定位模型权重，查看 `h2q_project/` 下的 `.pth` / `.pt` 文件名。

示例片段（参考实现风格）
- 查找接口索引：`report, index = generate_interface_map("./h2q_project")`。
- 启动本地 life-cycle：`docker run --rm --name h2q_life_cycle -v {PROJECT_ROOT}:/app/h2q_project -w /app/h2q_project h2q-sandbox python3 -u tools/heartbeat.py`。

保留事项（不要自动更改）
- 不要随意删除或重命名 `h2q_project/` 下的权重文件或实验数据文件（例如 `h2q_memory.pt`，`h2q_model_*`）。
- 不要改变 `evo_state.json` 的基本结构而不同时提供迁移逻辑。

如果有不清楚的点
- 请告知希望补充的具体区域（例如某个模块的运行示例、额外的启动命令或更多代码引用），我会按需求扩充或合并已有文档。

---
最后更新: 自动生成（基于仓库代码扫描） — 如需合并旧版本或包含额外示例，请回复要保留的文本。
