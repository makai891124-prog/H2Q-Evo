#!/bin/bash

echo "🔍 AGI进化系统自动监控守护进程启动..."
echo "=========================================="

while true; do
  echo "================================================ $(date) ================================================"

  # 检查进程状态
  echo "🔄 进程状态:"
  TRAINING_PID=$(pgrep -f deepseek_enhanced_agi_evolution.py)
  MONITOR_PID=$(pgrep -f agi_system_monitor.sh)

  if [ -n "$TRAINING_PID" ]; then
    echo "  ✅ 训练进程运行中 (PID: $TRAINING_PID)"
  else
    echo "  ❌ 训练进程未运行 - 重新启动..."
    nohup python3 deepseek_enhanced_agi_evolution.py > training_output.log 2>&1 &
    sleep 5
  fi

  if [ -n "$MONITOR_PID" ]; then
    echo "  ✅ 监控进程运行中 (PID: $MONITOR_PID)"
  else
    echo "  ❌ 监控进程未运行 - 重新启动..."
    nohup ./agi_system_monitor.sh > monitor_output.log 2>&1 &
    sleep 5
  fi

  # 检查模型状态
  echo ""
  echo "🤖 DeepSeek模型状态:"
  ollama list 2>/dev/null | grep deepseek | head -3

  # 检查存储使用
  echo ""
  echo "💾 存储状态:"
  du -sh . 2>/dev/null | awk '{print "  项目目录: " $1}'
  df -h / 2>/dev/null | tail -1 | awk '{print "  系统磁盘: " $4 " 可用"}'

  # 检查最新训练活动
  echo ""
  echo "📈 最新训练活动:"
  tail -3 agi_evolution_training.log 2>/dev/null || echo "  暂无训练日志"

  # 检查系统负载
  echo ""
  echo "⚡ 系统负载:"
  uptime 2>/dev/null | awk '{print "  负载: " $10 " " $11 " " $12}'

  # 检查训练统计
  echo ""
  echo "📊 训练统计:"
  if [ -f "evo_state.json" ]; then
    GENERATION=$(grep -o '"generation": [0-9]*' evo_state.json 2>/dev/null | grep -o '[0-9]*' | tail -1)
    if [ -n "$GENERATION" ]; then
      echo "  当前世代: $GENERATION"
    fi
  fi

  echo ""
  echo "⏱️  下次检查: 5分钟后..."
  sleep 300
done