#!/bin/bash
# H2Q-Evo AGI持久化训练启动脚本

set -e

echo "🚀 启动H2Q-Evo AGI持久化训练和进化系统"
echo "========================================"

# 检查Python环境
echo "📋 检查Python环境..."
python3 --version
pip --version

# 检查必要的依赖
echo "📦 检查依赖..."
python3 -c "
import torch
import transformers
import accelerate
import peft
import trl
import wandb
print('✅ 所有依赖已安装')
"

# 创建必要的目录
echo "📁 创建工作目录..."
mkdir -p agi_persistent_training/{checkpoints,logs,data}

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export WANDB_PROJECT="h2q-evo-persistent-agi"
export WANDB_WATCH="all"

# 显示配置信息
echo "⚙️  系统配置:"
echo "   基础模型: microsoft/DialoGPT-medium"
echo "   训练目标: 长期持续学习和进化"
echo "   内存限制: 8GB"
echo "   进化间隔: 24小时"
echo "   最大代数: 1000"

echo ""
echo "🧠 启动AGI持久化训练..."
echo "   按Ctrl+C可安全停止训练"
echo ""

# 启动训练
python3 agi_persistent_evolution.py

echo ""
echo "✅ 训练完成或已停止"