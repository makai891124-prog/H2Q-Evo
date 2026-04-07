# DAS 自主计算库说明（Autocompute Guide）

## 1. 目标
本指南定义 `h2q_project/das_gqs/autocompute.py` 的统一调用规范，用于在本地硬件上自动降低 DAS 计算摩擦。

覆盖能力：
1. 硬件探测与执行计划生成。
2. 设备-核-线程联合探针与自动调优。
3. staged rotor 预缓存索引执行。
4. 编译与精度策略的可控降级。

## 2. 模块位置
1. 实现：`h2q_project/das_gqs/autocompute.py`
2. 对外导出：`h2q_project/das_gqs/__init__.py`
3. 集成样例：`h2q_project/das_gqs/das_geometric_conversion_experiment.py`

## 3. 核心接口
### 3.1 硬件画像
```python
from h2q_project.das_gqs import detect_hardware_profile, profile_as_dict

profile = detect_hardware_profile()
print(profile_as_dict(profile))
```

输出字段：
1. `torch_version`
2. `cpu_count`
3. `torch_threads`
4. `has_cuda`
5. `has_mps`
6. `cuda_device_name`

### 3.2 设备选择
```python
from h2q_project.das_gqs import resolve_device

dev = resolve_device("auto")  # auto/cpu/mps/cuda
```

策略：
1. `auto` 优先级：`cuda -> mps -> cpu`。
2. 不可用设备会自动回退。

### 3.3 转子核选择
```python
from h2q_project.das_gqs import choose_rotor_kernel

kernel = choose_rotor_kernel("auto", device=dev, rank=128, rotor_steps=32)
```

核类型：
1. `scalar`：逐步逐对更新。
2. `staged`：同一 stage 内并行更新非冲突通道对。

### 3.4 线程候选与解析
```python
from h2q_project.das_gqs import default_thread_candidates, parse_thread_candidates

cand_default = default_thread_candidates(cpu_count=10)      # e.g. [1,2,4,6,8,10]
cand_custom = parse_thread_candidates("1,2,4,8", cpu_count=10)
```

## 4. staged 索引预缓存
### 4.1 设计
在模型初始化阶段完成：
1. `build_nonoverlap_stages(rotor_pairs)` 生成 stage 拓扑。
2. `pack_stage_steps(stages, device)` 打包为张量：
   - `stage_steps`: `[num_stages, max_len]`（padding=-1）
   - `stage_lengths`: `[num_stages]`

### 4.2 效果
避免每次 forward 动态构造 `idx tensor`，降低 Python/allocator 开销，尤其对小 batch 高频推理有利。

## 5. 自动调优准则
### 5.1 调优维度
1. `device`：cpu/mps/cuda
2. `kernel`：scalar/staged
3. `threads`：仅 CPU 路径扫描

### 5.2 推荐流程
1. 先做 `device × threads × kernel` 粗粒度探针，确定全局最优计划。
2. 再在固定计划上做 `kernel` 微调（可选）。
3. 若结果设备是 MPS，默认禁用 compile。

### 5.3 记录要求
每次实验报告必须写回：
1. `hardware_profile`
2. `compute_plan`
3. `compute_plan.selected_torch_threads`
4. `compute_plan.autotune_timings_ms`

## 6. 集成规范（建议）
### 6.1 统一 CLI 开关
建议在所有 DAS benchmark 脚本中统一支持：
1. `--compute-device {auto,cpu,mps,cuda}`
2. `--rotor-kernel {auto,scalar,staged}`
3. `--autotune-threads`
4. `--thread-candidates 1,2,4,6,8,10`
5. `--autotune-kernel`
6. `--compile`

### 6.2 线程恢复
任何线程探针都必须在 `finally` 中恢复原始线程数，避免污染后续实验。

## 7. 与数学结构体系的关系
1. 自主计算库不改变 DAS 数学结构（路径系数 + rotor 链 + 黎曼球面目标）。
2. 库只优化“执行映射”：让同一数学结构在本机硬件上以更低摩擦实现。
3. 因此它是“结构不变、执行自适应”的工程层。

## 8. 最小复现示例
```bash
/Users/imymm/H2Q-Evo/.venv/bin/python -m h2q_project.das_gqs.das_geometric_conversion_experiment \
  --in-features 1024 \
  --out-features 1024 \
  --rank 128 \
  --rotor-steps 32 \
  --batch-size 128 \
  --repeat 120 \
  --warmup 25 \
  --refine-steps 140 \
  --structure-lambda 0.20 \
  --interaction-lambda 0.08 \
  --compute-device auto \
  --rotor-kernel auto \
  --autotune-threads \
  --thread-candidates 1,2,4,6,8,10 \
  --autotune-kernel \
  --baseline-profile spectral_decay
```

## 9. 后续扩展建议
1. 增加 `thread + batch-size` 联合调优表（离线缓存）。
2. 为 `staged` 路径增加 C++/Triton fused kernel。
3. 增加 MPS 专用分块 gather/scatter 路径，减少图分割。
4. 输出跨脚本共享的 `compute_plan_cache.json`，实现热启动。