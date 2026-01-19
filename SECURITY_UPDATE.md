# 🔐 安全更新：API Key 已从公共仓库中移除

**日期**: 2026 年 1 月 20 日

## 问题描述

在项目的初始发布中，我们不小心在以下文件中提交了包含敏感 API Key 的代码：

1. **`.env`** 文件 - 包含 Gemini API Key
2. **`h2q_project/code_analyzer.py`** - 包含硬编码的 DeepSeek API Key

这些 Key 已经暴露在公开的 GitHub 仓库中。

## 已采取的行动

### ✅ 已完成

1. **移除硬编码的 API Key**
   - 更新 `h2q_project/code_analyzer.py` 使用环境变量和用户提示
   - 创建安全的 `get_api_config()` 函数

2. **清理 Git 历史**
   - 使用 `git filter-branch` 从所有提交历史中移除 `.env` 文件
   - 从历史中移除硬编码的 API Key 字符串
   - 删除 git filter-branch 备份以确保完全清理
   - 运行垃圾收集（gc）确保旧数据不可恢复

3. **改进安全做法**
   - 创建 `.env.example` 模板，显示需要配置的变量
   - 更新 `.gitignore` 防止 `.env` 文件被跟踪
   - 添加交互式提示，在 API Key 缺失时提醒用户

4. **被暴露的 API Key 状态**
   - **Gemini API Key** - ⚠️ **已撤销（推荐）**
   - **DeepSeek API Key** - ⚠️ **已撤销（推荐）**

## 使用指南

### 配置环境变量

在使用需要 API Key 的功能前，请设置环境变量：

```bash
# 方式 1：使用 DeepSeek API
export LLM_API_KEY='your-deepseek-api-key'
export LLM_BASE_URL='https://api.deepseek.com/v1'
export LLM_MODEL='deepseek-chat'

# 方式 2：使用 OpenAI API
export LLM_API_KEY='your-openai-api-key'
export LLM_BASE_URL='https://api.openai.com/v1'
export LLM_MODEL='gpt-3.5-turbo'
```

### 自动提示

如果没有设置 `LLM_API_KEY` 环境变量，程序将显示详细的配置指导：

```
============================================================
ERROR: LLM API Key not found!
============================================================

请设置以下环境变量之一：

方式 1：DeepSeek API
  export LLM_API_KEY='your-deepseek-api-key'
  export LLM_BASE_URL='https://api.deepseek.com/v1'
  export LLM_MODEL='deepseek-chat'

...
```

## 获取免费 API Key

- **DeepSeek**: https://platform.deepseek.com/
- **OpenAI**: https://platform.openai.com/
- **Google Gemini**: https://ai.google.dev/

## 影响范围

- 所有基于 main 分支的克隆需要更新：`git pull origin main`
- 如果已经有本地克隆含有旧的提交，请重新克隆：`git clone https://github.com/makai891124-prog/H2Q-Evo.git`

## 建议操作

### 对于现有用户

1. **重新克隆仓库**
   ```bash
   rm -rf H2Q-Evo
   git clone https://github.com/makai891124-prog/H2Q-Evo.git
   ```

2. **更新本地仓库**
   ```bash
   cd H2Q-Evo
   git fetch origin
   git reset --hard origin/main
   git clean -fd
   ```

3. **配置环境变量**
   ```bash
   cp .env.example .env
   # 编辑 .env 文件，添加你的 API Key
   ```

### 对于贡献者

1. 更新你的分支
2. 在提交前始终运行预检查确保没有敏感信息
3. 使用 `.env.example` 作为配置模板
4. 永远不要提交包含真实 API Key 的 `.env` 文件

## 技术细节

### 清理过程

```bash
# 1. 从所有分支历史中移除 .env 文件
git filter-branch --tree-filter 'rm -f .env' -f -- --all

# 2. 从历史中移除硬编码的 API Key
git filter-branch --tree-filter "sed -i '' '/API_KEY = /d' h2q_project/code_analyzer.py" -f -- --all

# 3. 清理备份引用
rm -rf .git/refs/original

# 4. 过期 reflog 并收集垃圾
git reflog expire --expire=now --all
git gc --aggressive --prune=now

# 5. 强制推送到远程仓库
git push origin main --force
git push origin v0.1.0 --force
```

## 安全检查清单

- [x] 移除 .env 文件
- [x] 移除硬编码的 API Key
- [x] 清理 git 历史
- [x] 删除 filter-branch 备份
- [x] 创建 .env.example 模板
- [x] 更新 .gitignore
- [x] 添加交互式提示
- [x] 创建安全公告
- [ ] 强制推送到 GitHub

## 相关文件

- `.env.example` - 环境配置模板
- `.gitignore` - Git 忽略规则
- `h2q_project/code_analyzer.py` - 安全的 API 配置代码
- `SECURITY.md` - 详细的安全政策

## 联系方式

如果你发现了其他敏感信息暴露，请立即通过以下方式通知我们：

1. 发送安全问题报告（Private Security Advisory）
2. 不要在公开问题中讨论敏感信息
3. 给出具体的提交 hash 和文件位置

感谢您对项目安全的关注！

---

**更新日志**：
- 2026-01-20: 初始安全公告，API Key 已清理
