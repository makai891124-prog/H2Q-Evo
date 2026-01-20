# H2Q-Evo 目录结构概览 (v2.3.0)

面向协作者的快速导览，覆盖主要代码、脚本、文档与数据位置。

## 代码主体
- h2q_cli/ — CLI 六命令入口
  - main.py, commands.py, config.py
- h2q_project/ — 核心工程目录
  - local_executor.py, learning_loop.py, strategy_manager.py, feedback_handler.py
  - knowledge/knowledge_db.py — SQLite 知识库
  - persistence/checkpoint_manager.py — 检查点/迁移/校验
  - monitoring/metrics_tracker.py — 指标追踪
  - h2q_server.py — FastAPI 推理服务
  - run_experiment.py / quick_experiment.py — 示例与快速实验
  - h2q/ — 四元数/分形核心库；*.pth, *.pt 预训练权重
- tests/ — 单元测试 (14+ 用例)，覆盖 CLI、本地执行、持久化
- tools/ — 辅助脚本，含 smoke_cli.py 烟雾测试

## 文档与报告
- README.md — 主入口，含索引与快速开始
- README_V2_3_0.md — v2.3.0 用户指南
- PRODUCTION_DEPLOYMENT_GUIDE_V2_3_0.md — 生产部署
- ACCEPTANCE_REPORT_V2_3_0.md — 验收报告
- FINAL_PROJECT_DELIVERY_SUMMARY.md — 最终交付总结
- GITHUB_COMMIT_SUMMARY_V2_3_0.md — GitHub 提交摘要
- docs/DOCUMENTATION_INDEX.md — 文档索引
- docs/STRUCTURE_OVERVIEW.md — 本文件
- 其他历史/专题报告 (根目录 .md) 保留以兼容已有链接

## 运行与验收
- validate_v2_3_0.py — E2E 验收脚本 (18/18 检查)
- production_acceptance_report.py — 生产验收报告生成
- tools/smoke_cli.py — CLI 烟雾测试
- acceptance_test.sh — 辅助验收脚本

## 配置与依赖
- pyproject.toml — 构建与入口点 (h2q = h2q_cli.main:main)
- requirements.txt / requirements_v2_3_0.txt — 依赖
- .env.example — 环境变量样例

## 日志与数据
- logs/ — 运行/训练日志
- training_checkpoints/ — 训练检查点 (如存在)
- temp_sandbox/ — 临时文件夹
- h2q_evolution_dataset.jsonl, learning_report*.json — 数据/报告输出

## 脚本与维护
- evolution_system.py, evolution_supervisor.py — 演化/调度
- project_graph.py — 模块索引工具
- reset_*.py / fix_*.py / diagnose_system.py — 运维与修复脚本
- publish_opensource.sh, run_agi_system.sh, start_agi.sh — 发布/启动脚本

## 兼容性说明
- 未移动或删除任何历史文件；所有旧路径保持有效。
- 若需进一步归档，可按需将不常用文档迁移至 docs/ 子目录，并更新索引。