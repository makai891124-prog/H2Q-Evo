#!/bin/bash
# AGI进化系统状态监控脚本

echo "🚀 H2Q-Evo AGI进化系统 - 7*24小时持续学习监控"
echo "================================================"
echo "开始时间: $(date)"
echo ""

while true; do
    echo "📊 系统状态检查 - $(date '+%H:%M:%S')"
    echo "----------------------------------------"

    # 检查进程状态
    echo "🔄 运行进程:"
    ps aux | grep python | grep -v grep | grep -E "(deepseek|monitor)" | while read line; do
        pid=$(echo $line | awk '{print $2}')
        cmd=$(echo $line | awk '{print $11}')
        echo "  PID $pid: $cmd"
    done

    # 检查DeepSeek模型状态
    echo ""
    echo "🤖 DeepSeek模型状态:"
    ollama list 2>/dev/null | grep deepseek | while read line; do
        model=$(echo $line | awk '{print $1}')
        size=$(echo $line | awk '{print $3, $4}')
        status=$(echo $line | awk '{print $5, $6, $7}')
        echo "  ✅ $model ($size) - $status"
    done

    # 检查存储使用情况
    echo ""
    echo "💾 存储监控:"
    du -sh /Users/imymm/H2Q-Evo 2>/dev/null | awk '{print "  项目目录:", $1}'
    df -h / | tail -1 | awk '{print "  系统磁盘:", $4, "可用"}'

    # 检查最新训练活动
    echo ""
    echo "📈 最新训练活动:"
    tail -3 /Users/imymm/H2Q-Evo/agi_evolution_training.log 2>/dev/null | while read line; do
        echo "  $line"
    done

    echo ""
    echo "⏱️  下次检查: 60秒后..."
    echo "================================================"
    sleep 60
done