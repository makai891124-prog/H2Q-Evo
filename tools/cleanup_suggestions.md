# Cleanup & Refactor Suggestions — H2Q-Evo

目标：在不破坏功能的前提下精简冗余文件、合并重复实现、并确保主运行路径集中使用统一数学核心。

重要原则
- 任何删除前先在分支上备份并运行 `tools/unified_audit.py` 与测试套件。
- 首选“归档”（移动到 `archive/`）而非直接删除，以便回滚。
- 逐步执行：先移除明显遗留或重复的文件，再合并功能重叠模块。

建议清理候选（按优先级）

1) 明确移除/归档（安全、已验证）
- `h2q_project/h2q_server_refactored.py` (已移除)
- `test_refactored_server.py` (已移除)

2) 高优先级（建议归档后运行完整审计再删除）
- `example_module.py`, `example_usage.py`, `module1.py` — 示例/占位模块，非生产。
- `train_model_a.py`, `train_model_b.py`, `train_simple_lstm.py`, `train_tpq_optim.py` — 多个训练样例实现重复，可合并为 `h2q_project/benchmarks/` 下统一训练器。
- `test_*.py` 文件散落在根目录（例如 `test_refactored_server.py` 已处理）— 将真正的单元测试保留在 `tests/` 并移除根目录的临时测试脚本。

3) 中优先级（需人工评估依赖）
- `local_inference_demo.py`, `local_long_text_generator.py`, `local_model_advanced_training.py` — 若功能被 `h2q_project` 中的新服务覆盖，可归档。
- `upgrade_super_env*`, `reset_docker*`, `fix_*` 脚本：保留但集中到 `tools/maintenance/`。

4) 低优先级/建议重构
- `train_spacetime*`, `train_fractal.py`, `fractal*`：将 fractal/training 代码整合到 `h2q_project/benchmarks/`，并增加清晰的入口参数。
- 把所有 `DiscreteDecisionEngine` 的多个实现合并（`automorphic_dde.py`, `train_fdc_pure.py`, `train_fdc.py`）保留一个 canonical 实现在 `h2q/core/discrete_decision_engine.py`。

自动化步骤（示例命令）

# 1) 归档文件到 archive/ 并运行统一审计
mkdir -p archive
git mv example_module.py archive/
python3 tools/unified_audit.py

# 2) 运行完整测试与基准（轻量）
PYTHONPATH=. pytest -q
PYTHONPATH=. python3 tools/unified_audit.py
PYTHONPATH=. MPLBACKEND=Agg python3 h2q_project/run_experiment.py

# 3) 完整基准（可选，耗时）
PYTHONPATH=. python3 deep_performance_audit.py --run-cifar10

变更建议呈报
- 我们应分三阶段执行清理：识别/归档 → 审计/测试 → 最终删除。
- 我可以按批次自动归档一小组候选文件、运行 `tools/unified_audit.py` 并提供差异报告（建议每批 5-10 文件）。

下一步（推荐）
- 允许我自动归档第一批高优先级候选（`example_module.py`, `module1.py`, 若存在）并运行统一审计；我会提交变更并报告结果。

