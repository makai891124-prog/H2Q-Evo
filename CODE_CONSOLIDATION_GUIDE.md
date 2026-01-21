# 项目代码合并工具使用指南

## 📌 已生成的文件

### 1. **PROJECT_CODE_CONSOLIDATED.md** (433.32 MB)
- **用途**: 完整的项目代码合并，包含所有源代码、配置、脚本等
- **文件数**: 19,311 个
- **代码行数**: 11,464,009 行
- **适用场景**: 
  - 需要全面了解项目结构的深度分析
  - AI 工具进行全局逻辑一致性检查
  - 完整的项目审计

### 2. **PROJECT_CORE_CODE_SUMMARY.md** (16.40 MB)
- **用途**: 核心业务代码的精简汇总，重点关注 `h2q_project` 目录
- **文件数**: 1,137 个核心文件
- **代码行数**: 3,603,697 行
- **适用场景**:
  - 快速理解项目核心逻辑
  - AI 工具进行核心架构分析
  - 团队代码审查和讨论
  - 性能更优，加载更快

## 🛠️ 工具脚本

### consolidate_project.py
```bash
python3 consolidate_project.py
```
生成完整的项目代码合并文件。可根据需要修改以下配置：
- `IGNORE_DIRS`: 忽略的目录列表
- `IGNORE_FILES`: 忽略的文件模式
- `CODE_EXTENSIONS`: 包含的代码文件扩展名

### consolidate_core_code.py
```bash
python3 consolidate_core_code.py
```
生成核心代码精简版本。专注于 `h2q_project` 等关键目录。

## 📊 文件结构

### 完整版特点
```
- 头部总览（生成时间、路径、统计信息）
- 📑 文件清单与导航（按文件类型分组）
- 📄 详细代码内容（每个文件单独标记）
- 📊 项目总结统计（代码类型分布表）
```

### 精简版特点
```
- 头部概览
- 📋 目录导航（按文件夹分类）
- 📝 源代码详情（前200行截断）
- 📊 代码统计
```

## 🎯 使用建议

### 对于 AI 工具分析

#### 使用完整版（PROJECT_CODE_CONSOLIDATED.md）
```
场景: 我需要分析整个项目的逻辑一致性
优势: 包含所有代码细节，可进行深度分析
劣势: 文件较大（433MB），处理成本高
```

#### 使用精简版（PROJECT_CORE_CODE_SUMMARY.md）
```
场景: 我需要快速理解项目核心架构
优势: 文件小（16.4MB），加载快，聚焦核心
劣势: 某些代码被截断到200行
```

### 手动编辑和扩展

#### 修改扫描范围
编辑 `consolidate_project.py` 中的以下变量：

```python
# 要忽略的目录
IGNORE_DIRS = {
    ".git", "node_modules", "test", ...
}

# 要包含的代码扩展名
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ...
}
```

#### 修改输出格式
调整 `get_language_for_extension()` 方法来支持更多代码语言的高亮。

## 💡 最佳实践

### 1. 前期使用精简版
```bash
# 1. 先用精简版快速理解架构
cat PROJECT_CORE_CODE_SUMMARY.md | head -n 1000

# 2. 用 AI 工具分析核心逻辑
# 提示词示例：分析这个项目的核心模块结构和依赖关系
```

### 2. 需要时切换到完整版
```bash
# 1. 若需要查看特定的细节代码
grep -n "function_name" PROJECT_CODE_CONSOLIDATED.md

# 2. 进行全局一致性检查
# 用 AI 分析完整版文件进行深度验证
```

### 3. 分割大文件（可选）
如果 433MB 的完整版太大，可以分割：

```bash
# 按行数分割
split -l 1000000 PROJECT_CODE_CONSOLIDATED.md PROJECT_PART_

# 按大小分割
split -b 100M PROJECT_CODE_CONSOLIDATED.md PROJECT_PART_
```

## 📈 性能参考

| 文件 | 大小 | 行数 | 文件数 | 加载时间* |
|------|------|------|--------|----------|
| 完整版 | 433 MB | 11.46M | 19,311 | ~30s |
| 精简版 | 16.4 MB | 3.60M | 1,137 | ~2s |

*估算值，取决于系统性能和工具

## 🔍 故障排除

### 文件生成失败
```bash
# 检查 Python 版本
python3 --version  # 需要 3.6+

# 检查磁盘空间
df -h .

# 查看详细错误
python3 consolidate_project.py 2>&1 | tail -n 50
```

### 文件打开困难
```bash
# 使用流式查看（避免一次加载整个文件）
less PROJECT_CODE_CONSOLIDATED.md

# 查看特定部分
head -n 5000 PROJECT_CODE_CONSOLIDATED.md > PREVIEW.md

# 搜索特定内容
grep -n "import" PROJECT_CORE_CODE_SUMMARY.md | head -n 20
```

## 📝 文件元信息

### 排除的文件类型
- 二进制文件: `.pyc`, `.pyo`, `.so`, `.dll`, `.exe`
- 模型文件: `.pt`, `.pth`, `.h5`, `.pb`, `.ckpt`
- 锁文件: `.lock`, `.yarn.lock`, `package-lock.json`
- 日志文件: `.log`, `evolution.log`
- 依赖目录: `node_modules`, `__pycache__`, `.venv`
- VCS 目录: `.git`, `.gitignore`

### 包含的文件类型
- Python: `.py`
- JavaScript/TypeScript: `.js`, `.ts`, `.jsx`, `.tsx`
- 配置: `.json`, `.yaml`, `.yml`, `.toml`, `.ini`, `.cfg`
- 脚本: `.sh`, `.bash`
- 数据库: `.sql`
- 标记: `.html`, `.css`

## 🚀 下一步建议

1. **选择合适的版本**
   - 初次分析 → 精简版
   - 深度审查 → 完整版

2. **与 AI 工具集成**
   - ChatGPT: 直接粘贴内容或上传文件
   - Claude: 同样支持大文件上传
   - 本地 LLM: 使用精简版节省资源

3. **持续维护**
   - 定期重新生成（如代码有较大变化）
   - 可加入到 CI/CD 流程中
   - 作为文档的补充资料

---

**最后更新**: 2026-01-21
**生成工具版本**: 1.0
