# H2Q-Evo（Reality-First 版）

> 目标：以“真实性第一”为原则，说明本仓库**真实可用的能力边界**与**已知失败/局限**，供社区评估与分析。

## 结论先行

- 本仓库**不等于 AGI**，也**没有达到稳定的通用推理能力**。
- 核心代码主要是**数学结构与工程化近似**的实验集合，**不是完备的模型训练或推理系统**。
- 多数“能力报告/验收报告/完成报告”属于**研究记录与产出整理**，不构成独立、可复现的强证据。

---

## 当前系统状态（2026-03-06）

以下状态基于仓库当前主流程代码与自动化脚本整理，便于开源协作方快速判断可用性与风险。

1) 进化与门禁主链路
- 已形成“自进化守护链路”：`capability_registry`、`public_alignment_report`、`nightly_regression_guard`、`release_gate`。
- 引入动态蓝图引导器：`tools/dynamic_blueprint_bootstrap.py`，可按当前能力缺口动态调度行动模块。
- 强门禁模式下支持“失败 -> 恢复 -> 自动降阈值重试”的闭环策略（用于提升收敛稳定性）。

2) 测试与 CI 状态
- 已加入针对动态蓝图的聚焦测试文件：`tests/test_dynamic_blueprint_bootstrap.py`。
- CI 工作流中增加定向测试步骤，使用 `-o addopts=''` 避开仓库全局 pytest addopts/cov 干扰。
- 该聚焦测试在本地验证通过（4/4）。

3) 运行模式边界
- API 模式依赖外部模型与密钥配置，缺失时会退化到本地模板/回退路径。
- Local + Docker 路径依赖本地 Docker daemon、镜像与容器内路径一致性；缺失会导致一致性门禁失败。
- 报告文件高频生成且数量大，建议将时间戳产物视为运行工件而非核心源码。

4) 当前建议的开源协作姿态
- 将“可复现脚本 + 最小测试”作为合并优先级高于“结论性文档”。
- 对外宣称能力以 `release_gate`、回归守护与公开可运行脚本为准。
- 以 `*_latest.*` 报告作为运行快照入口，避免被历史时间戳产物淹没。

---

## 真实可用的模块（基于代码审计）

以下模块在代码层面**可运行/可调用**，但多为**工程近似**，不代表严格数学或模型性能保证。

### 1) DAS 核心框架（Directional Axiomatic System）
- Z2 与正交扩展群、度量不变性、构造宇宙与群作用的抽象实现。
- 主要用途：数学结构实验与报告输出，不是训练或推理主干。
- 入口：h2q_project/das_core.py

### 2) 四元数 + SU(2) 基础算子
- Hamilton 乘法、共轭、归一化、指数/对数映射等基础运算。
- 主要用途：数学实验与状态变换。
- 入口：h2q_project/h2q/core/lie_automorphism_engine.py

### 3) 分形缩放与 IFS 迭代
- Hausdorff 维度缩放、多层 IFS 迭代、分形导数的数值近似。
- 入口：h2q_project/h2q/core/lie_automorphism_engine.py

### 4) 非交换几何与反射算子（近似）
- Fueter 导数、反射微分、Weyl 反射、Ricci flow 的近似实现。
- 入口：h2q_project/h2q/core/noncommutative_geometry_operators.py

### 5) 纽结不变量（参数化近似）
- Alexander/Jones/HOMFLY 多项式以“可学习系数 + 幂级数”方式实现。
- Khovanov 同调用统计分级近似实现。
- 入口：h2q_project/h2q/core/knot_invariant_hub.py

### 6) 谱位移与拓扑撕裂检测（η 指标）
- 通过 $\eta = \frac{1}{\pi}\arg\det(S)$ 进行“拓扑一致性”评估。
- 入口：h2q_project/h2q/core/automorphic_dde.py、h2q_project/src/h2q/core/engine.py、h2q_project/src/h2q/core/sst.py

### 7) 统一数学架构（模块融合）
- 将李群、分形、非交换几何、纽结约束、DDE 组合为统一前向。
- 入口：h2q_project/h2q/core/unified_architecture.py、h2q_project/h2q/core/evolution_integration.py

### 8) 记忆系统（示范级）
- 使用任务哈希生成“伪语义四元数”，再用点积检索。
- **不是 embedding 或语义向量**，属于演示性质。
- 入口：h2q_project/fractal_memory.py

### 9) 工具合成器（需 LLM 才能生效）
- 依赖外部 LLM（API key）生成工具代码并本地测试验证。
- 无 LLM 时无法发挥作用。
- 入口：h2q_project/tool_synthesizer.py

### 10) 精度门控执行器（验证层）
- 以熵、命题对偶和四元数语义编码控制输出可信度。
- 主要作用：守护/验证层，而非生成模型。
- 入口：h2q_project/precision_gated_executor.py

### 11) FastAPI 服务（工程演示）
- /chat 与 /generate 接口存在，但输出是张量投影或简化解码。
- **不等价于成熟的语言模型服务**。
- 入口：h2q_project/h2q_server.py

---

## 已知失败与局限（真实性优先）

1) **缺乏可复现的通用能力**
- 公开基准与内部评估显示能力偏低，无法达到“可靠通用”标准。

2) **“数学结构”多为近似或占位**
- 纽结理论、非交换几何等属于工程近似，不是严格数学实现。

3) **记忆系统不是真正的语义检索**
- 当前基于哈希随机四元数，不具备真实语义表达能力。

4) **推理服务不等于 LLM**
- /chat 与 /generate 的输出是张量投影与简化解码，并非语言模型生成。

5) **报告与验收文档不具备独立可验证性**
- 许多报告来自项目内部生成，可能缺乏第三方复现证据。

---

## 快速验证（仅用于结构检查）

> 这些命令仅验证“代码可运行路径”，不代表真实能力。

1) 启动 API（演示用）
- PYTHONPATH=. python3 -m uvicorn h2q_project.h2q_server:app --reload --host 0.0.0.0 --port 8000

2) 启动进化系统（演示/日志）
- python3 evolution_system.py

---

## 目录导览（建议阅读顺序）

1) h2q_project/das_core.py
2) h2q_project/h2q/core/lie_automorphism_engine.py
3) h2q_project/h2q/core/noncommutative_geometry_operators.py
4) h2q_project/h2q/core/knot_invariant_hub.py
5) h2q_project/h2q/core/unified_architecture.py
6) h2q_project/h2q/core/evolution_integration.py
7) h2q_project/h2q_server.py
8) h2q_project/fractal_memory.py
9) h2q_project/tool_synthesizer.py
10) h2q_project/precision_gated_executor.py

---

## 适合谁阅读

- 研究者：希望了解“数学结构 + 工程化近似”如何被组织成实验系统。
- 工程师：评估哪些模块可复用、哪些只是概念验证。
- 审计者：需要明确“可运行”与“可宣称能力”之间的边界。

---

## 免责声明

- 本仓库**不保证任何 AGI 能力**。
- 本仓库的报告、验收文档、演示脚本不构成可重复的性能承诺。
- 如果你需要严谨评测与可复现结果，请以独立基准与外部复现为准。

---

## 贡献方式

如果你想基于现实边界改进系统：
- 优先提交可复现的基准与测评脚本。
- 在 PR 中提供最小可复现示例（MRE）。
- 明确标注“演示/概念验证/可复现实验”的类型。

---

**版本**：Reality-First README（2026-03-06）
