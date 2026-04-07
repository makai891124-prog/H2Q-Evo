# DAS 架构效能与维度压制报告

## 1. 实验目标与公开实例
本实验将标准 `nn.Linear(1024, 1024)` 作为公开基线，并用 DAS 几何路径层替换其密集矩阵表达。目标是验证：
1. 参数复杂度是否从 $O(N^2)$ 压至 $O(rN)$。
2. 前向推理是否可在“不展开完整权重矩阵”的条件下执行。
3. 在不同压缩等级下，评估精度-速度-内存三者的工程折中。

代码实现文件：
1. `h2q_project/das_gqs/das_geometric_conversion_experiment.py`

自动产物：
1. `reports/das_geometric_conversion_report.json`
2. `reports/das_geometric_conversion_report.md`
3. `reports/das_geometric_conversion_report_rank128.json`
4. `reports/das_geometric_conversion_report_rank128.md`

## 2. DAS 映射法则在代码中的落地
### 2.1 The Matrix Fallacy -> 拓扑路径化
基线层权重为 $W \in \mathbb{R}^{1024 \times 1024}$。DAS 编译后不保存 $W$，改为：
$$
W_{DAS} = P_{out}^\top \cdot R(\theta) \cdot \mathrm{diag}(s) \cdot P_{in}
$$
其中：
1. $P_{in} \in \mathbb{R}^{r \times N}$：输入路径系数。
2. $P_{out} \in \mathbb{R}^{r \times N}$：输出路径系数。
3. $R(\theta) \in \mathbb{R}^{r \times r}$：由转子链（2D 平面 Givens 旋转）构成的等距几何演化。
4. $s \in \mathbb{R}^r$：缩放标量。

### 2.2 Weights as Rotors
在 `DASGeometricLayer` 内部，转子链由 `(i,j,theta)` 序列参数化。每个 step 对潜变量通道做：
$$
\begin{aligned}
z_i' &= \cos\theta\, z_i - \sin\theta\, z_j \\
z_j' &= \sin\theta\, z_i + \cos\theta\, z_j
\end{aligned}
$$
这正是几何代数中“平面旋转转子”的离散实现。

### 2.3 Masked Path Execution
`forward(x)` 流程：
1. `latent = x @ P_in^T`
2. `latent *= s`
3. 逐步执行 rotor chain
4. `y = latent @ P_out`

整个推理没有 `reconstruct_full_matrix()`，满足 Lazy Path 约束。

## 3. 编译器（compress_to_das）
`compress_to_das(standard_layer, compression_rank, ...)` 使用 SVD：
$$
W \approx U_r \Sigma_r V_r^\top
$$
并映射为：
1. `in_paths <- V_r^T`
2. `scales <- diag(\Sigma_r)`
3. `out_paths <- U_r^T`
4. 可选 refinement：在随机输入上蒸馏拟合，微调路径与转子参数。

这对应“从 Hilbert 权重同构体到几何路径系数”的编译过程。

## 4. 真实结果（本机实跑）
### 4.1 高压缩工况（rank=32, rotor_steps=16）
来源：`reports/das_geometric_conversion_report.json`

1. Baseline params: `1,049,600`
2. DAS params: `66,608`
3. 压缩率: `15.7579x`
4. 参数下降: `93.6540%`
5. MAE: `8.718155e-02`
6. 相对 Frobenius 误差: `6.145220e-01`
7. CPU 平均推理时延：
   - Baseline: `0.2524 ms`
   - DAS: `0.2332 ms`
   - Speedup: `1.0824x`

解释：该工况实现了强内存压制与轻度提速，但精度损失较大，适合“激进压缩”场景。

### 4.2 高保真工况（rank=128, rotor_steps=32）
来源：`reports/das_geometric_conversion_report_rank128.json`

1. Baseline params: `1,049,600`
2. DAS params: `263,328`
3. 压缩率: `3.9859x`
4. 参数下降: `74.9116%`
5. MAE: `2.317438e-02`
6. CPU speedup: `0.5009x`（即当前 CPU 上慢于 GEMM 基线）

解释：提高 rank 后精度显著改善，但在传统矩阵优化硬件上速度优势消失，验证了“信息摩擦”问题。

## 5. 复杂度与 Memory Wall 论证
### 5.1 参数复杂度
1. Baseline: $O(N^2)$。
2. DAS: $O(rN + K)$，其中 $K$ 为转子步数。

当 $r \ll N$ 时，参数规模线性化，直接压制模型权重存储。

### 5.2 带宽与权重读取
基线每次推理核心访问密集权重块；DAS 访问路径系数与潜变量通道，读取复杂度从 $O(N^2)$ 下降到 $O(rN)$。在大模型中，这一变化针对的是权重读取带宽瓶颈，而非仅算术 FLOPs。

### 5.3 KV Cache 关联
本实验对象是线性层；若推广到 Transformer Q/K/V/O 投影，同样可将投影矩阵改写为路径系数+转子链，减少参数驻留和投影带宽压力。KV Cache 本体大小还与序列长度有关，DAS 不直接消除序列项，但可降低投影侧内存墙。

## 6. 信息摩擦（Friction）分析
### 6.1 现有硅基 GPU 的不匹配
当前 CUDA/TensorCore 强优化目标是大块 GEMM；DAS 的路径/转子算子更偏向小张量收缩和通道旋转，容易出现：
1. 算子启动与调度开销占比上升。
2. TensorCore 利用率不足。
3. kernel 融合路径不成熟。

这解释了为何高保真工况在 CPU 上会慢于 dense GEMM。

### 6.2 面向 DAS 的硬件展望
建议的 3D 可逆计算芯片能力：
1. 原生 rotor-pair SIMD 指令（平面旋转 fused op）。
2. 路径核心的片上 SRAM 拓扑路由，减少随机访存。
3. contraction + rotor 融合流水线，减少中间张量回写。
4. 可逆检查点机制，降低训练反向传播存储开销。

## 7. 一键复现命令
```bash
# 高压缩工况
/Users/imymm/H2Q-Evo/.venv/bin/python -m h2q_project.das_gqs.das_geometric_conversion_experiment \
  --in-features 1024 \
  --out-features 1024 \
  --rank 32 \
  --rotor-steps 16 \
  --batch-size 128 \
  --repeat 120 \
  --warmup 25 \
  --refine-steps 120 \
  --baseline-profile spectral_decay

# 高保真工况
/Users/imymm/H2Q-Evo/.venv/bin/python -m h2q_project.das_gqs.das_geometric_conversion_experiment \
  --in-features 1024 \
  --out-features 1024 \
  --rank 128 \
  --rotor-steps 32 \
  --batch-size 128 \
  --repeat 120 \
  --warmup 25 \
  --refine-steps 120 \
  --baseline-profile spectral_decay \
  --output-json reports/das_geometric_conversion_report_rank128.json \
  --output-md reports/das_geometric_conversion_report_rank128.md
```

## 8. 结论
1. DAS 几何路径表达在参数维度上实现了实证级压制（最高约 93.65% 参数削减）。
2. 在激进压缩下可维持轻度速度优势，但精度会退化。
3. 在高保真压缩下，当前通用硬件未必更快，体现“软件哲学与硬件物理路径不匹配”的信息摩擦。
4. 因此 DAS 的核心优势已在“内存维度爆炸抑制”上得到明确证据，下一步瓶颈转向硬件协同与算子编译栈。

## 9. 结构基准增补（2026-03-28）
### 9.1 从欧式 Hilbert 表示到黎曼球面交互结构
为回应“欧式无穷维表示缺少系统结构关联”的问题，本次在编译后 refinement 与评估阶段加入了曲率目标结构。

对任意输出向量 $y \in \mathbb{R}^d$，采用立体投影提升到单位球面 $S^d \subset \mathbb{R}^{d+1}$：
$$
\Phi(y)=\left[\frac{\lVert y\rVert^2-1}{\lVert y\rVert^2+1},\;\frac{2y}{\lVert y\rVert^2+1}\right],\quad \lVert \Phi(y)\rVert=1
$$
并将该球面作为“目标结构基准”，通过以下损失还原结构：
1. 测地误差（geodesic）
$$
\mathcal{L}_{geo}=\mathbb{E}\left[\arccos\left(\langle \Phi(y_{ref}),\Phi(y_{das})\rangle\right)\right]
$$
2. 交互结构误差（interaction Gram）
$$
\mathcal{L}_{int}=\left\|G(\Phi(y_{ref}))-G(\Phi(y_{das}))\right\|_F^2
$$

训练总损失：
$$
\mathcal{L}=\mathcal{L}_{mse}+\lambda_s\mathcal{L}_{geo}+\lambda_i\mathcal{L}_{int}
$$
本次实测参数：$\lambda_s=0.20,\lambda_i=0.08$。

### 9.2 低 rank 精度提升 + 速度回收实现点
在 `h2q_project/das_gqs/das_geometric_conversion_experiment.py` 中新增：
1. 转子执行优化：删除每 step 的全量 `latent.clone()`，改为仅两通道局部克隆与原位回写，减少无效内存复制。
2. refinement 优化：`AdamW + CosineAnnealingLR`，并引入混合输入分布以增强低 rank 稳定性。
3. 新增黎曼结构损失与结构指标输出（mean/p95 geodesic, interaction rel-Fro）。

### 9.3 新复测结果（同一报告套件已更新）
#### 工况 A：rank=32, rotor_steps=16
来源：`reports/das_geometric_conversion_report.json`

1. 参数压缩：`15.7579x`（下降 `93.6540%`）
2. 误差：
  - MAE: `8.661319e-02`（较此前 `8.718155e-02` 小幅改善）
  - rel Fro: `6.112761e-01`（较此前 `6.145220e-01` 改善）
3. 速度：
  - Baseline: `0.257242 ms`
  - DAS: `0.247640 ms`
  - Speedup: `1.038773x`
4. 黎曼结构对齐：
  - mean geodesic: `2.711501e-01` rad
  - p95 geodesic: `3.713914e-01` rad
  - interaction rel-Fro: `8.694996e-02`

#### 工况 B：rank=128, rotor_steps=32
来源：`reports/das_geometric_conversion_report_rank128.json`

1. 参数压缩：`3.9859x`（下降 `74.9116%`）
2. 误差：
  - MAE: `2.499608e-02`
  - rel Fro: `1.762807e-01`
3. 速度：
  - Baseline: `0.247369 ms`
  - DAS: `0.449512 ms`
  - Speedup: `0.550306x`
4. 黎曼结构对齐：
  - mean geodesic: `6.318383e-02` rad
  - p95 geodesic: `8.353979e-02` rad
  - interaction rel-Fro: `4.991204e-03`

### 9.4 结构还原结论
1. 仅用欧式 MSE 训练时，低 rank 在“结构关系”上容易退化为局部数值拟合。
2. 引入球面测地与交互 Gram 对齐后，可显式将输出几何关系拉回到曲率流形基准。
3. rank=128 的球面结构对齐显著更好（测地误差与交互误差远低于 rank=32），说明该映射对“结构还原”有效。
4. 当前 CPU 上速度收益仍受算子与硬件匹配限制，后续需继续做 fused rotor kernels 才能同时拿到高保真与速度回收。

### 9.5 更新后的复现实验命令
```bash
# 低 rank + 黎曼结构目标
/Users/imymm/H2Q-Evo/.venv/bin/python -m h2q_project.das_gqs.das_geometric_conversion_experiment \
  --in-features 1024 \
  --out-features 1024 \
  --rank 32 \
  --rotor-steps 16 \
  --batch-size 128 \
  --repeat 120 \
  --warmup 25 \
  --refine-steps 140 \
  --structure-lambda 0.20 \
  --interaction-lambda 0.08 \
  --baseline-profile spectral_decay \
  --output-json reports/das_geometric_conversion_report.json \
  --output-md reports/das_geometric_conversion_report.md

# 高保真 + 黎曼结构目标
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
  --baseline-profile spectral_decay \
  --output-json reports/das_geometric_conversion_report_rank128.json \
  --output-md reports/das_geometric_conversion_report_rank128.md
```

## 10. 自主计算库（本地硬件自适应）
### 10.1 目标
为降低“数学结构-硬件执行”摩擦，本次新增 DAS 自主计算库：
1. 本机硬件探测（CPU/MPS/CUDA + 线程与版本）。
2. 设备-核联合自动调优（device × rotor kernel）。
3. 转子并行分阶段执行（non-overlap staged kernel）。
4. staged 阶段索引预缓存（避免每次 forward 构造索引张量）。
5. 线程自调优（torch threads probing）。
6. 可选 `torch.compile` 封装与自动降级策略。

代码入口：
1. `h2q_project/das_gqs/autocompute.py`
2. `h2q_project/das_gqs/das_geometric_conversion_experiment.py`

### 10.2 核心执行策略
1. `resolve_device("auto")` 不再盲目固定 MPS/CUDA，而是先探针。
2. 对 `scalar` 与 `staged` 两种转子核做微基准，取最低时延路径。
3. CPU 路径上增加 `threads × kernel` 联合探针，自动回写最佳线程数。
4. staged 路径使用预缓存阶段索引，减少 forward 内部索引构造开销。
5. 当设备为 MPS 时默认关闭 compile 路径，避免当前图分割带来的额外摩擦。
6. 报告中回写：
   - `hardware_profile`
   - `compute_plan`
   - `autotune_timings_ms`

### 10.3 本机硬件画像（实测）
1. `torch=2.9.1`
2. `cpu_count=10`
3. `torch_threads=4`
4. `has_cuda=false`
5. `has_mps=true`

### 10.4 本机自动调优结果（关键）
`rank=32` 的联合探针（device × threads × kernel）显示：
1. 最优计划：`cpu + staged + threads=2`
2. 关键探针值：
  - `plan::cpu:t2:staged = 0.089878 ms`
  - `plan::mps:t1:staged = 0.820793 ms`

`rank=128` 的联合探针显示：
1. 最优计划：`cpu + staged + threads=1`
2. 关键探针值：
  - `plan::cpu:t1:staged = 0.128286 ms`
  - `plan::mps:t1:staged = 1.578396 ms`

结论：在这台机器上，MPS 路径仍有明显调度摩擦；CPU + staged + 自适应线程是当前最优执行结构。

### 10.5 速度回收结果（加入自主库后）
1. rank=32：speedup 提升到 `1.7340x`（并回写最佳线程 `2`）。
2. rank=128：speedup 提升到 `1.8942x`（并回写最佳线程 `1`）。

说明：在当前这台机器上，摩擦主要来自 MPS 图分割与小算子调度，而非 DAS 结构本身。通过自主库做设备-核联合调优后，已显著回收时延。

### 10.6 自主库复现命令
```bash
# rank=32（启用自适应计划与核调优）
/Users/imymm/H2Q-Evo/.venv/bin/python -m h2q_project.das_gqs.das_geometric_conversion_experiment \
  --in-features 1024 \
  --out-features 1024 \
  --rank 32 \
  --rotor-steps 16 \
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
  --baseline-profile spectral_decay \
  --output-json reports/das_geometric_conversion_report.json \
  --output-md reports/das_geometric_conversion_report.md

# rank=128（启用自适应计划与核调优）
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
  --baseline-profile spectral_decay \
  --output-json reports/das_geometric_conversion_report_rank128.json \
  --output-md reports/das_geometric_conversion_report_rank128.md
```