# H2Q-Evo AGI完整训练系统

这是一个完整的AGI训练基础设施，集成了所有必要的训练前置组件，实现实时在线训练、热生成、连续操作和动态备份。

## 系统组件

### 1. AGI训练基础设施 (`agi_training_infrastructure.py`)
- **系统环境监控**: 实时监控CPU、内存、磁盘和网络状态
- **动态备份系统**: 自动备份系统状态，支持滚动备份和故障恢复
- **训练数据管道**: 结构化数据加载和预处理
- **热重载管理器**: 支持运行时更新组件
- **资源管理器**: 监控和控制系统资源使用
- **性能监控器**: 实时性能指标收集

### 2. AGI检查点系统 (`agi_checkpoint_system.py`)
- **模型检查点管理器**: 保存和恢复训练状态
- **版本管理**: 支持多版本检查点管理
- **回滚管理器**: 自动故障恢复和回滚
- **检查点导出/导入**: 支持检查点迁移

### 3. AGI容错系统 (`agi_fault_tolerance.py`)
- **故障类型检测**: 自动识别不同类型的系统故障
- **恢复策略**: 多种自动恢复策略（重启、回滚、重试等）
- **健康检查**: 持续监控系统健康状态
- **熔断器模式**: 防止故障级联
- **进程监督**: 自动重启崩溃的进程

### 4. AGI实时训练系统 (`agi_realtime_training.py`)
- **实时训练**: 持续的在线训练过程
- **热生成**: 运行时生成新的训练组件
- **环境感知**: 根据系统状态动态调整训练参数
- **连续操作**: 7×24小时不间断运行
- **动态备份**: 自动备份防止数据丢失

## 快速开始

### 1. 安装依赖

```bash
pip install torch numpy psutil asyncio
```

### 2. 检查系统要求

```bash
python start_agi_training.py --check-only
```

### 3. 启动完整训练系统

```bash
python start_agi_training.py
```

### 4. 监控训练状态

系统会自动创建以下监控文件：
- `agi_system_status.json`: 实时状态
- `agi_system_report.json`: 详细报告
- `realtime_training_status.json`: 训练状态

## 配置选项

### 命令行参数

- `--check-only`: 仅检查系统要求
- `--config <file>`: 指定配置文件
- `--log-level <level>`: 设置日志级别 (DEBUG, INFO, WARNING, ERROR)

### 环境变量

- `AGI_TRAINING_ENABLED`: 启用/禁用训练 (默认: true)
- `AGI_HOT_GENERATION_ENABLED`: 启用/禁用热生成 (默认: true)
- `AGI_CONTINUOUS_OPERATION`: 启用/禁用连续操作 (默认: true)
- `AGI_DYNAMIC_BACKUP_ENABLED`: 启用/禁用动态备份 (默认: true)
- `AGI_ENVIRONMENTAL_SENSING`: 启用/禁用环境感知 (默认: true)

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    AGI训练系统启动器                          │
│                    start_agi_training.py                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
           ┌──────────┴──────────┐
           │                     │
┌──────────▼─────────┐ ┌─────────▼──────────┐
│  实时训练系统      │ │   容错系统         │
│agi_realtime_training│ │agi_fault_tolerance│
└──────────┬─────────┘ └─────────┬──────────┘
           │                     │
           └──────────┬──────────┘
                      │
           ┌──────────▼──────────┐
           │                     │
┌──────────▼─────────┐ ┌─────────▼──────────┐
│  训练基础设施      │ │   检查点系统       │
│agi_training_infrastructure│ │agi_checkpoint_system│
└────────────────────┘ └────────────────────┘
```

## 监控和日志

### 日志文件
- `agi_training_system.log`: 主系统日志
- `evolution.log`: 训练演化日志
- `fault_alerts.txt`: 故障警报

### 状态文件
- `agi_system_status.json`: 系统运行状态
- `agi_system_report.json`: 详细系统报告
- `realtime_training_status.json`: 训练进度

### 备份文件
- `agi_backups/`: 系统状态备份
- `checkpoints/`: 模型检查点

## 故障恢复

系统内置多种故障恢复机制：

1. **自动重启**: 进程崩溃时自动重启
2. **检查点回滚**: 训练发散时回滚到稳定状态
3. **服务降级**: 资源不足时降低训练强度
4. **熔断保护**: 防止故障级联影响

## 性能优化

### 环境感知调节
- CPU使用率 > 80%: 减少批次大小
- 内存使用率 > 85%: 降低训练强度
- 网络断开: 暂停外部数据同步

### 资源管理
- 自动调整学习率
- 动态批次大小调节
- 内存使用优化

## 扩展开发

### 添加新的健康检查

```python
from agi_fault_tolerance import get_fault_tolerance_manager

ft_manager = get_fault_tolerance_manager()
ft_manager.register_health_check("custom_check", your_check_function, interval=60)
```

### 自定义恢复策略

```python
from agi_fault_tolerance import RecoveryStrategy, FaultToleranceManager

class CustomRecoveryManager(FaultToleranceManager):
    def _custom_recovery(self, fault_record):
        # 实现自定义恢复逻辑
        pass
```

### 集成新的训练组件

```python
from agi_realtime_training import H2QRealtimeTrainer

class CustomTrainer(H2QRealtimeTrainer):
    def _perform_training_step(self):
        # 实现自定义训练逻辑
        pass
```

## 安全注意事项

1. **数据备份**: 系统自动备份，但建议定期外部备份
2. **权限控制**: 确保运行用户有适当的文件系统权限
3. **网络安全**: 在生产环境中配置适当的网络安全措施
4. **资源限制**: 监控系统资源使用，避免影响其他服务

## 故障排除

### 常见问题

1. **模块导入错误**
   - 检查Python路径和依赖安装
   - 运行 `pip install -r requirements.txt`

2. **权限错误**
   - 检查文件和目录权限
   - 确保用户有读写权限

3. **内存不足**
   - 减少批次大小
   - 启用服务降级模式

4. **训练不收敛**
   - 检查数据质量
   - 调整学习率
   - 使用检查点回滚

### 调试模式

启用详细日志：

```bash
python start_agi_training.py --log-level DEBUG
```

## 贡献指南

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 发起Pull Request

## 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 支持

如有问题，请：
1. 查看日志文件
2. 检查系统状态文件
3. 提交Issue到项目仓库

---

**注意**: 这是一个复杂的AGI训练系统，请在有经验的开发者监督下运行。系统设计用于连续操作，包含自动恢复机制，但仍需要定期监控。