# H2Q-Evo AGI 自主可进化工程系统 - 快速开始指南

## 🎯 系统目标

**自主可进化的 AGI 工程系统，专注于科学领域的原理开发、解算和工程方法落地自组织。**

### 核心能力

- **数学**: 定理证明、优化求解、方程推导
- **物理**: 模型构建、方程求解、数值模拟
- **化学**: 反应机理分析、分子设计、量化计算
- **生物**: 系统建模、通路分析、结构预测
- **工程**: 有限元分析、多目标优化、设计方法

---

## 🚀 快速开始

### 方式1: 一键部署（推荐）

```bash
# 完整部署 (4小时训练)
python3 deploy_agi_final.py --hours 4 --download-data

# 快速测试 (30分钟)
python3 deploy_agi_final.py --hours 0.5 --download-data

# 长期训练 (12小时)
python3 deploy_agi_final.py --hours 12 --download-data

# 使用现有数据（跳过下载）
python3 deploy_agi_final.py --hours 4 --no-download
```

### 方式2: 分步执行

```bash
# 步骤1: 下载科学数据集
cd h2q_project
python3 scientific_dataset_loader.py

# 步骤2: 启动AGI训练
python3 agi_scientific_trainer.py \
  --data ./scientific_datasets/scientific_training_data.jsonl \
  --duration 4.0 \
  --output ./agi_training_output

# 步骤3: 查看结果
ls -lh agi_training_output/
cat agi_training_output/agi_training_report_*.md
```

---

## 📊 训练结果分析

### 查看演示训练结果

```bash
# 检查之前的30分钟演示训练结果
cat h2q_project/training_output/training_report.md
cat h2q_project/training_output/training_report.json
```

**演示训练统计** (已完成):
- **迭代次数**: 72,975 次
- **训练时长**: 30 分钟 (1,800 秒)
- **性能评分**: 100.0% (一致性维持)
- **状态**: ✅ 完成

### 输出文件位置

1. **科学数据集**:
   - 位置: `h2q_project/scientific_datasets/`
   - 文件: `scientific_dataset_*.json`
   - 训练数据: `scientific_training_data.jsonl`

2. **AGI训练结果**:
   - 位置: `h2q_project/agi_training_output/`
   - 报告: `agi_training_report_*.md`
   - 数据: `agi_training_results_*.json`

3. **日志文件**:
   - 训练日志: `h2q_project/agi_scientific_training.log`
   - 演示日志: `h2q_project/training_progress.log`

---

## 🔬 科学数据集详情

### 数据源

1. **arXiv 论文** (自动下载):
   - 数学: `math.CO`, `math.AG` (组合数学、代数几何)
   - 物理: `physics.comp-ph`, `chem-ph` (计算物理、化学物理)
   - 生物: `q-bio.BM` (生物分子)

2. **合成科学数据** (内置):
   - 数学问题与定理 (拉格朗日乘数法、不等式证明等)
   - 物理推导 (量子谐振子、麦克斯韦方程等)
   - 化学机理 (SN2反应、化学平衡等)
   - 生物过程 (蛋白质折叠、细胞呼吸等)
   - 工程方法 (有限元分析、优化设计等)

### 数据格式

训练数据采用 JSONL 格式:

```json
{
  "prompt": "请详细解答以下数学领域的问题：\n\n拉格朗日乘数法在约束优化中的应用",
  "response": "给定目标函数 f(x,y) = x² + y²，约束条件 g(x,y) = x + y - 1 = 0...",
  "metadata": {
    "domain": "mathematics",
    "source": "synthetic",
    "type": "problem"
  }
}
```

---

## 🧠 AGI 训练系统架构

### 核心组件

1. **ScientificKnowledgeBase** (科学知识库)
   - 分领域存储知识
   - 持续积累学习成果
   - 支持知识检索和推理

2. **ScientificReasoningEngine** (科学推理引擎)
   - 问题分析与分类
   - 复杂度评估
   - 推理策略选择
   - 逐步求解

3. **AGIScientificTrainer** (AGI训练器)
   - 迭代训练循环
   - 实时性能监控
   - 知识库更新
   - 进化数据记录

### 训练流程

```
加载数据 → 问题分析 → 策略选择 → 推理求解 → 知识积累 → 性能评估 → 迭代
    ↑                                                              ↓
    └──────────────────────────── 自组织进化 ─────────────────────┘
```

---

## 📈 性能监控

### 实时指标

- **总迭代次数**: 完成的训练迭代
- **解决问题数**: 成功求解的科学问题
- **覆盖领域数**: 涉及的科学领域数量
- **平均置信度**: 求解结果的平均置信度
- **知识库增长**: 各领域知识条目数量

### 示例输出

```
[迭代  5000] 进度: 25.0% | 已解决: 5000 | 领域: 5 | 剩余: 3h 0m 0s
[迭代 10000] 进度: 50.0% | 已解决: 10000 | 领域: 5 | 剩余: 2h 0m 0s
[迭代 15000] 进度: 75.0% | 已解决: 15000 | 领域: 5 | 剩余: 1h 0m 0s
```

---

## 🎓 系统能力演示

### 数学推理

**输入**: "使用拉格朗日乘数法求解约束优化问题"

**系统能力**:
1. 识别问题类型: 约束优化
2. 选择策略: 演绎推理 + 形式化证明
3. 推导步骤:
   - 构造拉格朗日函数
   - 求偏导数
   - 解方程组
   - 验证结果

### 物理建模

**输入**: "推导量子谐振子能级"

**系统能力**:
1. 识别: 量子力学本征值问题
2. 策略: 从基本原理推导
3. 步骤:
   - 写出哈密顿量
   - 引入升降算符
   - 求解本征方程
   - 得到能级公式

### 化学机理

**输入**: "分析SN2反应机理"

**系统能力**:
1. 识别: 有机反应机理
2. 策略: 反应路径分析
3. 分析:
   - 确定反应类型
   - 追踪电子流动
   - 预测中间态
   - 给出速率方程

---

## 🔄 进化机制

### 自组织学习

系统通过以下机制实现自主进化:

1. **知识积累**: 每次求解后更新知识库
2. **策略优化**: 根据成功率调整推理策略
3. **跨域迁移**: 不同领域的知识相互借鉴
4. **元学习**: 学习如何更好地学习

### 进化指标

- **知识库增长率**: 新知识条目/时间
- **求解成功率**: 成功求解/总尝试
- **置信度提升**: 置信度随时间的变化
- **领域覆盖度**: 覆盖的科学领域范围

---

## 🛠 高级配置

### 自定义训练参数

编辑 `h2q_project/agi_scientific_trainer.py`:

```python
# 修改训练参数
AGI_CONFIG = {
    "reasoning_depth": 5,        # 推理深度
    "knowledge_retention": 0.95, # 知识保留率
    "exploration_rate": 0.2,     # 探索率
    "confidence_threshold": 0.7,  # 置信度阈值
}
```

### 添加新的科学数据源

编辑 `h2q_project/scientific_dataset_loader.py`:

```python
DATASET_CONFIG = {
    "custom_source": {
        "enabled": True,
        "url": "https://...",
        "fields": ["title", "abstract", "keywords"],
        "max_results": 100,
    }
}
```

---

## 📝 使用案例

### 案例1: 数学定理证明训练

```bash
# 专注数学领域的长期训练
python3 deploy_agi_final.py --hours 12 --download-data

# 查看数学知识库
python3 -c "
import json
with open('h2q_project/agi_training_output/agi_training_results_*.json') as f:
    data = json.load(f)
    print('数学知识:', data['knowledge_base_stats']['mathematics'])
"
```

### 案例2: 物理模拟训练

```bash
# 下载更多物理论文
# 修改 scientific_dataset_loader.py 增加物理类别

# 启动训练
python3 h2q_project/agi_scientific_trainer.py \
  --data h2q_project/scientific_datasets/scientific_training_data.jsonl \
  --duration 8
```

### 案例3: 跨领域整合训练

```bash
# 24小时长期训练以促进跨领域知识整合
python3 deploy_agi_final.py --hours 24 --download-data
```

---

## 🔍 故障排除

### 问题1: 数据下载失败

**原因**: 网络问题或API限制

**解决**:
```bash
# 使用现有数据跳过下载
python3 deploy_agi_final.py --no-download

# 或手动下载后运行
python3 h2q_project/scientific_dataset_loader.py
```

### 问题2: 训练过慢

**原因**: 系统负载或数据量大

**解决**:
- 减少训练时长: `--hours 1`
- 减少数据量: 编辑配置文件减少 `max_results`
- 关闭其他应用释放资源

### 问题3: 内存不足

**解决**:
```bash
# 减少知识库大小（编辑代码）
# 或限制训练样本数量
```

---

## 📚 参考文档

- [长时间训练指南](LONG_TIME_TRAINING_GUIDE.md)
- [本地模型训练指南](h2q_project/LOCAL_MODEL_TRAINING_GUIDE.md)
- [发布说明 v2.2.0](RELEASE_NOTES_V2.2.0.md)
- [项目结构说明](.github/copilot-instructions.md)

---

## 🎯 下一步发展方向

### 短期目标 (1-3个月)

1. ✅ 科学数据集集成
2. ✅ AGI训练框架
3. 🔄 符号计算引擎集成
4. 🔄 方程自动推导
5. 🔄 可视化分析工具

### 中期目标 (3-6个月)

1. 多模态理解 (文本 + 图像 + 公式)
2. 自主实验设计系统
3. 知识图谱构建
4. 元学习能力
5. 分布式训练支持

### 长期愿景 (6-12个月)

1. 完全自主的科学研究助手
2. 跨领域原理发现
3. 工程方法自动落地
4. 自组织演化系统
5. 人机协作研究平台

---

## 🤝 贡献指南

欢迎贡献代码、数据集或改进建议！

### 贡献方式

1. Fork 项目
2. 创建特性分支
3. 提交改进
4. 发起 Pull Request

### 重点领域

- 新的科学数据源
- 领域特定的推理引擎
- 性能优化
- 可视化工具
- 文档改进

---

**版本**: v2.2.0+  
**最后更新**: 2026-01-20  
**维护**: H2Q-Evo Team  
**许可**: MIT License
