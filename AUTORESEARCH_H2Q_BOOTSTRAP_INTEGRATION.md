# Autoresearch x H2Q-Evo Fusion

## Goal
将 `karpathy/autoresearch` 的自动研究闭环思想融合到 H2Q-Evo 本地自举系统中：
- fixed-budget experiment loop
- keep/discard/crash ledger
- next-step self-improvement plan

## What Was Integrated
新增编排器：`tools/run_autoresearch_h2q_bootstrap.py`

该脚本映射关系：
- Autoresearch 的 `results.tsv` 经验池 -> `--autoresearch-results` 读取并提取 top KEEP 描述
- Autoresearch 的 keep/discard/crash 决策 -> 本地每个实验产出 status
- Autoresearch 的 loop forever 思路 -> `--max-iterations N` 循环调度（可夜间长跑）

## Local H2Q Experiment Targets
融合脚本会调度本地三个核心实验：
1. `tools/run_self_eval_distillation_pipeline.py`
2. `tools/run_research_aggregation_cross_validation.py`
3. `tools/run_systemic_platform_joint_capability_assessment.py --ci-safe`

指标映射：
- `delta_schema_valid_rate`
- `aggregate.score` (research aggregation)
- `aggregate.score` (systemic joint)

## Usage
规划模式（不执行命令，只产出计划与调度）：

```bash
/Users/imymm/H2Q-Evo/.venv/bin/python tools/run_autoresearch_h2q_bootstrap.py --max-iterations 4
```

执行模式（真正运行实验）：

```bash
/Users/imymm/H2Q-Evo/.venv/bin/python tools/run_autoresearch_h2q_bootstrap.py --execute --max-iterations 3 --timeout-sec 900
```

可指定 autoresearch 结果文件：

```bash
/Users/imymm/H2Q-Evo/.venv/bin/python tools/run_autoresearch_h2q_bootstrap.py --execute --autoresearch-results external/autoresearch/results.tsv
```

## Output Artifacts
- `reports/autoresearch_h2q_bootstrap_fusion_latest.json`
- `reports/autoresearch_h2q_bootstrap_fusion_latest.md`
- `reports/autoresearch_h2q_experiment_ledger_latest.tsv`

## Notes
- 本融合实现不执行 `git reset`，避免对当前仓库产生破坏性回退。
- `systemic joint` 使用 `--ci-safe`，减少 Lean/外部环境依赖带来的不稳定。
- 该方案是“研究闭环编排层”，可进一步接入 one-click 主流程。