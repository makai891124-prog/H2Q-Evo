#!/bin/bash
# H2Q-Evo AGI 完整系统启动脚本

echo "================================================================================"
echo "🚀 H2Q-Evo AGI 完整系统启动器"
echo "================================================================================"
echo ""

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装"
    exit 1
fi

echo "✓ Python3 已安装: $(python3 --version)"
echo ""

# 显示菜单
echo "请选择运行模式:"
echo ""
echo "  1) 🎬 演示模式 - 快速展示AGI能力（推荐）"
echo "  2) 💬 交互模式 - 与AGI对话"
echo "  3) 🤖 自动运行模式 - 自动执行多个查询"
echo "  4) 🎓 学习模式 - 运行智能学习系统"
echo "  5) 👁️  监控模式 - 查看系统状态（需要守护进程运行）"
echo "  6) 🔧 守护进程模式 - 后台持续运行"
echo "  7) 📊 知识库统计 - 查看知识库详情"
echo "  8) ✅ 一键完整演示 - 运行所有核心功能"
echo ""
read -p "选择 (1-8): " choice

case $choice in
    1)
        echo ""
        echo "▶️  启动演示模式..."
        python3 integrated_agi_system.py demo
        ;;
    2)
        echo ""
        echo "▶️  启动交互模式..."
        python3 integrated_agi_system.py
        ;;
    3)
        echo ""
        echo "▶️  启动自动运行模式..."
        python3 integrated_agi_system.py auto
        ;;
    4)
        echo ""
        echo "▶️  启动学习模式 (3周期, 每周期10项)..."
        python3 intelligent_learning_system.py 3 10 2
        ;;
    5)
        echo ""
        echo "▶️  启动监控面板..."
        if [ ! -f "agi_daemon_status.json" ]; then
            echo "⚠️  守护进程未运行，先启动守护进程（选项6）"
            exit 1
        fi
        python3 monitor_agi.py
        ;;
    6)
        echo ""
        echo "▶️  启动守护进程（后台运行，每10秒一次查询）..."
        nohup python3 agi_daemon.py 10 > agi_daemon.log 2>&1 &
        echo "✓ 守护进程已启动 (PID: $!)"
        echo "  查看日志: tail -f agi_daemon.log"
        echo "  监控状态: python3 monitor_agi.py"
        echo "  停止进程: pkill -f agi_daemon.py"
        ;;
    7)
        echo ""
        echo "▶️  显示知识库统计..."
        python3 large_knowledge_base.py
        ;;
    8)
        echo ""
        echo "================================================================================"
        echo "🎯 一键完整演示 - 展示所有核心功能"
        echo "================================================================================"
        echo ""
        
        echo "📚 步骤 1/4: 初始化知识库..."
        python3 large_knowledge_base.py | head -30
        echo ""
        sleep 2
        
        echo "🧠 步骤 2/4: 运行智能学习系统..."
        python3 intelligent_learning_system.py 2 10 1
        echo ""
        sleep 2
        
        echo "🤖 步骤 3/4: 运行集成AGI演示..."
        python3 integrated_agi_system.py demo
        echo ""
        sleep 2
        
        echo "📊 步骤 4/4: 显示最终状态..."
        python3 -c "
from large_knowledge_base import LargeKnowledgeBase
kb = LargeKnowledgeBase()
kb.load()
stats = kb.get_stats()
print('='*80)
print('📈 最终知识库统计')
print('='*80)
print(f'总知识: {stats[\"total_count\"]} 条')
print(f'已验证: {stats[\"verified_count\"]} 条 ({stats[\"verified_count\"]/max(stats[\"total_count\"],1)*100:.1f}%)')
print(f'未验证: {stats[\"unverified_count\"]} 条')
print('')
print('各领域验证率:')
for domain, total in sorted(stats['by_domain'].items()):
    verified = sum(1 for k in kb.knowledge[domain] if k.get('verified'))
    print(f'  {domain:20s}: {verified:2d}/{total:2d} ({verified/max(total,1)*100:.0f}%)')
print('='*80)
"
        echo ""
        echo "✅ 完整演示完成！"
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

echo ""
echo "================================================================================"
echo "✅ 完成"
echo "================================================================================"
