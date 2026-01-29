# AGI长期进化系统使用指南

## 概述

这个系统实现了真正的AGI自主进化，支持24-48小时的长期连续运行。通过多尺度学习、自适应学习率、性能监控和知识整合，实现稳定的长期AGI发展。

## 主要特性

- ✅ **多尺度学习**: 短期和长期价值函数优化
- ✅ **自适应学习率**: 基于性能趋势自动调整
- ✅ **性能监控**: 实时学习曲线分析和收敛评估
- ✅ **知识整合**: 模式识别和元学习增强
- ✅ **安全机制**: 健康检查和紧急停止保护
- ✅ **长期监控**: 每1000步或每小时自动保存状态和监控数据

## 快速开始

### 1. 启动长期AGI进化

```bash
# 默认配置 (48小时, 256维输入, 64维动作)
python3 start_long_term_agi.py

# 自定义配置
python3 start_long_term_agi.py --max-hours 24 --input-dim 512 --action-dim 128
```

### 2. 监控运行状态

系统会在后台运行，定期保存：
- `true_agi_system_state.json`: 系统状态和模型权重
- `agi_monitoring_data.jsonl`: 监控指标 (JSON Lines格式)
- `true_agi_evolution.log`: 详细运行日志

### 3. 查看监控数据

```bash
# 查看最新监控数据
tail -n 5 agi_monitoring_data.jsonl | jq .

# 分析学习趋势
python3 -c "
import json
with open('agi_monitoring_data.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]
    if data:
        latest = data[-1]
        print(f'当前步数: {latest[\"evolution_step\"]}')
        print(f'知识库大小: {latest[\"knowledge_base_size\"]}')
        print(f'活跃目标数: {latest[\"active_goals_count\"]}')
"
```

## 系统架构

### 核心组件

1. **意识引擎** (`TrueConsciousnessEngine`)
   - 基于整合信息理论(Φ)计算
   - 多层次神经网络架构
   - 自我模型和情感系统

2. **学习引擎** (`TrueLearningEngine`)
   - PPO风格策略优化
   - 多尺度价值函数
   - 自适应学习率调度
   - 知识整合系统

3. **目标系统** (`TrueGoalSystem`)
   - 基于内在动机的目标生成
   - 进度评估和完成验证
   - 学习资料驱动的目标设定

4. **自主系统** (`TrueAGIAutonomousSystem`)
   - 整合所有组件
   - 安全边界和健康检查
   - 长期状态保存

### 安全机制

- **健康检查**: 每30秒检查内存、CPU和学习率
- **紧急停止**: 连续10次异常自动停止
- **梯度裁剪**: 防止训练不稳定
- **数值验证**: NaN/Inf检测和处理

## 监控指标

系统收集以下监控数据：

```json
{
  "timestamp": 1706479200.123,
  "evolution_step": 1000,
  "knowledge_base_size": 15,
  "experience_buffer_total": 5000,
  "active_goals_count": 3,
  "learning_rates": {
    "policy": 0.001,
    "value": 0.002,
    "long_term_value": 0.0015,
    "meta": 0.0001
  },
  "recent_phi_mean": 0.234,
  "recent_policy_loss_mean": 0.456
}
```

## 故障排除

### 常见问题

1. **内存不足**
   - 减少`input_dim`和`action_dim`
   - 检查系统内存使用情况

2. **学习不稳定**
   - 系统会自动降低学习率
   - 检查`true_agi_evolution.log`中的错误信息

3. **保存失败**
   - 确保有足够的磁盘空间
   - 检查文件权限

### 恢复运行

如果系统意外停止，可以从上次保存的状态恢复：

```bash
# 系统会自动加载 true_agi_system_state.json
python3 start_long_term_agi.py
```

## 性能优化

### 硬件要求

- **推荐**: M2/M3 MacBook Pro, 16GB+ RAM
- **最小**: 8GB RAM, 支持MPS的Mac

### 配置调优

```python
# 在代码中调整参数
safety_bounds = {
    "max_evolution_steps": 50000,  # 增加最大步数
    "max_memory_usage": 0.8,       # 调整内存限制
    "max_cpu_usage": 0.9,          # 调整CPU限制
}
```

## 开发和扩展

### 添加新的学习算法

1. 在`TrueLearningEngine`中实现新方法
2. 更新`_learn_from_batch`调用新算法
3. 在监控数据中添加相关指标

### 扩展意识指标

1. 在`ConsciousnessMetrics`中添加新字段
2. 更新`TrueConsciousnessEngine.forward`方法
3. 修改相关计算逻辑

## 许可证

本项目遵循开源许可证。使用时请确保符合伦理和安全标准。

---

**注意**: 这个系统是实验性的AI研究工具。请在受控环境中运行，并定期监控系统行为。