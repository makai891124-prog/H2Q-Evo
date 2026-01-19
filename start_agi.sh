#!/bin/bash
# H2Q-Evo 本地量子AGI - 一键启动脚本

set -e

echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║                                                                   ║"
echo "║         H2Q-Evo 本地量子AGI生命体 - 快速启动                      ║"
echo "║                                                                   ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到python3"
    echo "请先安装Python 3.8或更高版本"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo "✓ 发现Python $PYTHON_VERSION"

# 检查numpy
if python3 -c "import numpy" 2>/dev/null; then
    echo "✓ NumPy已安装"
else
    echo "⚠️  警告: NumPy未安装"
    echo "正在安装NumPy..."
    pip3 install numpy --quiet || {
        echo "❌ 安装失败，请手动运行: pip3 install numpy"
        exit 1
    }
    echo "✓ NumPy安装完成"
fi

# 检查PyTorch（可选）
if python3 -c "import torch" 2>/dev/null; then
    echo "✓ PyTorch已安装（增强模式）"
    HAS_TORCH=true
else
    echo "ℹ️  PyTorch未安装（将使用模拟模式）"
    HAS_TORCH=false
fi

# 检查模型文件
MODEL_COUNT=$(find h2q_project -name "*.pth" -o -name "*.pt" 2>/dev/null | wc -l)
echo "✓ 发现 $MODEL_COUNT 个模型文件"

echo ""
echo "选择运行模式:"
echo "  1) 终端交互版（推荐，无GUI依赖）"
echo "  2) 基础GUI版（需要tkinter）"
echo "  3) 增强GUI版（需要tkinter + torch）"
echo "  4) 自动演示（运行示例查询）"
echo ""

# 如果是自动化脚本调用，使用参数
if [ $# -eq 1 ]; then
    CHOICE=$1
else
    read -p "请选择 [1-4]: " CHOICE
fi

case $CHOICE in
    1)
        echo ""
        echo "启动终端交互版..."
        python3 TERMINAL_AGI.py
        ;;
    2)
        echo ""
        echo "启动基础GUI版..."
        if python3 -c "import tkinter" 2>/dev/null; then
            python3 LOCAL_QUANTUM_AGI_LIFEFORM.py
        else
            echo "❌ 错误: tkinter未安装"
            echo "请参考 LOCAL_AGI_GUIDE.md 安装tkinter"
            exit 1
        fi
        ;;
    3)
        echo ""
        echo "启动增强GUI版..."
        if ! $HAS_TORCH; then
            echo "❌ 错误: PyTorch未安装"
            echo "请运行: pip3 install torch"
            exit 1
        fi
        if python3 -c "import tkinter" 2>/dev/null; then
            python3 ENHANCED_LOCAL_AGI.py
        else
            echo "❌ 错误: tkinter未安装"
            echo "请参考 LOCAL_AGI_GUIDE.md 安装tkinter"
            exit 1
        fi
        ;;
    4)
        echo ""
        echo "运行自动演示..."
        cat << 'DEMO' | python3 TERMINAL_AGI.py
prove 量子纠缠不变性
quantum 分析三量子比特GHZ态的拓扑不变量
models
status
exit
DEMO
        echo ""
        echo "✅ 演示完成！"
        echo ""
        echo "尝试手动运行获得交互体验:"
        echo "  ./start_agi.sh 1"
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac
