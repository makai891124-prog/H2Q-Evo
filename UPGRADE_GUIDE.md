# H2Q-Evo v2.1.0 升级和部署指南

**发布日期**: 2026-01-20  
**版本**: v2.1.0 Production Ready  
**适用平台**: macOS, Linux, Docker

---

## 📋 升级前检查清单

### 前置条件

- [ ] Python 3.10+ 已安装
- [ ] Git 已安装并配置
- [ ] 网络连接良好
- [ ] 磁盘空间 > 2GB
- [ ] Docker (可选，用于容器部署)

### 备份和准备

- [ ] **备份当前生产数据**
  ```bash
  cp -r /path/to/h2q_project ~/backup_h2q_$(date +%Y%m%d)
  ```

- [ ] **备份数据库和配置**
  ```bash
  cp -r /path/to/configs ~/backup_configs_$(date +%Y%m%d)
  ```

- [ ] **记录当前版本和性能指标**
  ```bash
  git describe --tags > current_version.txt
  python -m pytest tests/ --tb=short > test_results_before.txt
  ```

- [ ] **通知相关团队**
  - 更新日志
  - 部署时间表
  - 回滚计划

---

## 🚀 升级步骤

### 方法 1: 标准升级 (推荐)

```bash
# 1. 进入项目目录
cd /Users/imymm/H2Q-Evo

# 2. 拉取最新代码
git fetch origin
git checkout main
git pull origin main

# 3. 检查新版本标签
git tag -l | grep v2.1

# 4. 安装/更新依赖
pip install -r h2q_project/requirements.txt --upgrade

# 5. 运行系统检查
PYTHONPATH=/Users/imymm/H2Q-Evo/h2q_project \
python h2q_project/system_analyzer.py

# 6. 运行健康检查
PYTHONPATH=/Users/imymm/H2Q-Evo/h2q_project \
python h2q_project/h2q/core/production_validator.py

# 7. 运行测试
cd h2q_project
python -m pytest tests/ -v

# 8. 运行完整演示
python production_demo.py
```

### 方法 2: Docker 升级

```bash
# 1. 构建新镜像
docker build -t h2q-evo:v2.1.0 .

# 2. 创建新容器（带数据卷）
docker run -d \
  --name h2q-prod-v2.1.0 \
  -p 8000:8000 \
  -v h2q-data:/app/data \
  -e INFERENCE_MODE=local \
  h2q-evo:v2.1.0

# 3. 验证容器状态
docker ps
docker logs h2q-prod-v2.1.0

# 4. 测试健康检查端点
curl http://localhost:8000/health

# 5. 验证成功后移除旧容器
docker stop h2q-prod-old
docker rm h2q-prod-old
```

### 方法 3: 从源代码编译

```bash
# 1. 克隆仓库
git clone https://github.com/makai891124-prog/H2Q-Evo.git
cd H2Q-Evo

# 2. 检出版本标签
git checkout v2.1.0

# 3. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# 或 venv\Scripts\activate  # Windows

# 4. 安装依赖
pip install -r h2q_project/requirements.txt

# 5. 运行初始化
cd h2q_project
python -c "from h2q.core.production_validator import ProductionValidator; ProductionValidator().run_full_validation()"
```

---

## ✅ 升级验证

### 快速验证

```bash
# 1. 检查版本
cd /Users/imymm/H2Q-Evo
git describe --tags

# 2. 验证新文件存在
ls -la h2q_project/system_analyzer.py
ls -la h2q_project/h2q/core/algorithm_version_control.py
ls -la h2q_project/h2q/core/production_validator.py
ls -la h2q_project/h2q/core/robustness_wrapper.py

# 3. 运行快速测试
python -m pytest h2q_project/tests/test_dde.py -v

# 4. 验证性能
PYTHONPATH=/Users/imymm/H2Q-Evo/h2q_project \
python h2q_project/production_demo.py
```

### 完整验证

```bash
# 1. 系统健康检查
python h2q_project/h2q/core/production_validator.py

# 2. 代码分析
python h2q_project/system_analyzer.py

# 3. 全部测试
pytest h2q_project/tests/ -v --cov=h2q_project

# 4. 性能基准
python h2q_project/production_demo.py

# 5. 内存检查
python -c "
import tracemalloc
from h2q.core.discrete_decision_engine import get_canonical_dde, LatentConfig
tracemalloc.start()
config = LatentConfig(latent_dim=256, n_choices=64)
model = get_canonical_dde(config)
current, peak = tracemalloc.get_traced_memory()
print(f'Current: {current / 1024 / 1024:.1f}MB')
print(f'Peak: {peak / 1024 / 1024:.1f}MB')
"
```

### 预期结果

✅ **升级成功标志**:

```
✅ 版本标签: v2.1.0
✅ 新文件完整
✅ 所有测试通过 (22/22 PASSED)
✅ 系统健康状态: HEALTHY
✅ 推理延迟: < 2ms
✅ 内存使用: < 250MB
✅ 性能吞吐: > 800 QPS
```

---

## 🔄 分阶段部署策略

### 灰度发布 (推荐)

#### Phase 1: 金丝雀部署 (Week 1-2)

**目标**: 5%-50% 流量

```bash
# 1. 部署到灰度池
docker tag h2q-evo:v2.1.0 h2q-evo:canary
docker push h2q-evo:canary

# 2. 配置负载均衡器
# 设置 5% 流量到新版本
# 修改 nginx/k8s 配置

# 3. 监控指标
# - 延迟 (< 2ms)
# - 错误率 (< 0.1%)
# - CPU 使用
# - 内存使用
# - QPS

# 4. 收集反馈
# - 用户报告
# - 监控告警
# - 错误日志
```

#### Phase 2: 扩展部署 (Week 3-4)

**目标**: 50%-100% 流量

```bash
# 1. 验证金丝雀指标正常
# (完成 Phase 1 验证)

# 2. 扩展到 50% 流量
# 修改负载均衡器配置

# 3. 继续监控 24 小时

# 4. 全量切换到 100%
docker tag h2q-evo:v2.1.0 h2q-evo:latest
docker push h2q-evo:latest

# 5. 更新所有服务指向最新版本
```

#### Phase 3: 优化和稳定 (Week 5+)

**目标**: 系统稳定运行

```bash
# 1. 性能优化
# - 补充测试覆盖
# - 完善错误处理
# - 调整资源配置

# 2. 监控集成
# - 集成 Prometheus
# - 配置 Grafana 面板
# - 设置告警规则

# 3. 文档完善
# - API 文档更新
# - 运维手册
# - 故障排查指南
```

---

## 🔙 回滚步骤

**如果需要回滚到之前的版本**:

### 快速回滚

```bash
# 1. 停止当前版本
docker stop h2q-prod-v2.1.0
docker rm h2q-prod-v2.1.0

# 2. 启动前一个版本
docker run -d \
  --name h2q-prod-v2.0.x \
  -p 8000:8000 \
  -v h2q-data:/app/data \
  h2q-evo:v2.0.x

# 3. 验证版本
curl http://localhost:8000/health

# 4. 更新负载均衡器指向旧版本

# 5. 验证流量恢复正常
```

### 代码回滚

```bash
# 1. 查看提交历史
git log --oneline -10

# 2. 回滚到上一个稳定版本
git checkout v2.0.x

# 3. 重新部署
docker build -t h2q-evo:v2.0.x .
```

### 数据回滚

```bash
# 1. 停止当前应用
systemctl stop h2q-service

# 2. 恢复备份数据
cp -r ~/backup_h2q_* /path/to/h2q_project

# 3. 重启应用
systemctl start h2q-service
```

---

## 📊 关键监控指标

### 实时监控项

```bash
# 持续监控这些指标，确保升级成功

# 1. 推理性能
PYTHONPATH=/Users/imymm/H2Q-Evo/h2q_project \
watch -n 5 'curl -s http://localhost:8000/metrics | grep inference_latency'

# 2. 错误率
tail -f evolution.log | grep ERROR

# 3. QPS 吞吐
tail -f evolution.log | grep "requests processed"

# 4. 内存使用
watch -n 5 'ps aux | grep h2q_server | grep -v grep'

# 5. 健康状态
curl -s http://localhost:8000/health | jq .
```

### 监控仪表板

推荐集成以下监控系统：

- **Prometheus**: 指标收集
- **Grafana**: 可视化仪表板
- **ELK Stack**: 日志分析
- **Jaeger**: 分布式追踪

---

## 🆘 故障排查

### 常见问题

**问题 1: 升级后推理性能下降**

```bash
# 原因: 可能是 GPU 内存不足

# 解决方案:
export CUDA_VISIBLE_DEVICES=""  # 禁用 GPU
# 或调整批处理大小
python h2q_project/production_demo.py
```

**问题 2: 内存使用过高**

```bash
# 原因: 内存泄漏或数据累积

# 解决方案:
# 1. 重启应用
systemctl restart h2q-service

# 2. 清理临时文件
rm -rf /tmp/h2q_*

# 3. 检查日志
tail -100 evolution.log
```

**问题 3: 某些测试失败**

```bash
# 原因: 依赖版本冲突

# 解决方案:
pip install -r h2q_project/requirements.txt --force-reinstall
python -m pytest h2q_project/tests/test_dde.py -v --tb=short
```

**问题 4: Docker 容器无法启动**

```bash
# 原因: 镜像构建失败或配置问题

# 解决方案:
# 1. 检查构建日志
docker build -t h2q-evo:v2.1.0 . --verbose

# 2. 检查容器日志
docker logs h2q-prod-v2.1.0

# 3. 检查磁盘空间
df -h

# 4. 重新构建镜像
docker build --no-cache -t h2q-evo:v2.1.0 .
```

### 获取帮助

```bash
# 1. 查看最近的错误
tail -50 evolution.log | grep -i error

# 2. 运行诊断
python h2q_project/diagnose_system.py

# 3. 收集诊断信息
python h2q_project/audit_project_structure.py > diagnostic_report.txt

# 4. 提交 Issue
# https://github.com/makai891124-prog/H2Q-Evo/issues/new
```

---

## 📞 技术支持

### 联系方式

- 📧 **Email**: support@h2q-evo.dev
- 💬 **GitHub Issues**: https://github.com/makai891124-prog/H2Q-Evo/issues
- 💡 **Discussions**: https://github.com/makai891124-prog/H2Q-Evo/discussions
- 📖 **文档**: https://docs.h2q-evo.dev

### 相关文档

- [PRODUCTION_READINESS_REPORT.md](h2q_project/reports/PRODUCTION_READINESS_REPORT.md)
- [SYSTEM_HEALTH_REPORT.md](h2q_project/reports/SYSTEM_HEALTH_REPORT.md)
- [RELEASE_NOTES_V2.1.0.md](RELEASE_NOTES_V2.1.0.md)
- [CHANGELOG.md](CHANGELOG.md)

---

## ✅ 升级完成确认

升级完成后，请完成以下确认：

- [ ] 版本验证成功 (v2.1.0)
- [ ] 所有测试通过 (22/22)
- [ ] 系统健康检查通过 (HEALTHY)
- [ ] 性能指标符合预期
- [ ] 用户反馈良好
- [ ] 监控告警无异常
- [ ] 备份数据完整
- [ ] 文档已更新
- [ ] 团队已培训
- [ ] 后续改进计划已制定

---

**升级日期**: ________________  
**升级人员**: ________________  
**验证状态**: ________________  
**备注**: ________________________________________________________________

---

*本指南由 H2Q-Evo 团队维护*  
*最后更新: 2026-01-20*
