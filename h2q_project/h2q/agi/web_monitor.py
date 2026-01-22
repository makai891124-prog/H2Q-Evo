#!/usr/bin/env python3
"""
AGI训练监控Web面板
Web-based AGI Training Monitor Dashboard

在浏览器中实时查看训练进度
"""

import os
import json
import time
import subprocess
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional

try:
    from flask import Flask, render_template_string, jsonify
except ImportError:
    print("正在安装Flask...")
    os.system("pip3 install flask -q")
    from flask import Flask, render_template_string, jsonify

import torch

# ============================================================
# 配置
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
LOG_FILE = SCRIPT_DIR / 'optimized_training.log'
CHECKPOINT_DIR = SCRIPT_DIR / 'optimized_checkpoints'
MODEL_DIR = SCRIPT_DIR / 'optimized_models'

app = Flask(__name__)

# ============================================================
# HTML模板 - 现代化监控面板
# ============================================================

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤖 AGI训练监控面板</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            color: #e8e8e8;
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            padding: 20px;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.5em;
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        
        .header .time {
            color: #888;
            font-size: 1.1em;
        }
        
        .status-bar {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .status-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 20px;
            background: rgba(255,255,255,0.1);
            border-radius: 20px;
        }
        
        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        .status-dot.running {
            background: #00ff88;
            box-shadow: 0 0 10px #00ff88;
        }
        
        .status-dot.stopped {
            background: #ff4444;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .card {
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
        }
        
        .card h2 {
            font-size: 1.2em;
            margin-bottom: 20px;
            color: #00d9ff;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 12px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        
        .metric:last-child {
            border-bottom: none;
        }
        
        .metric-label {
            color: #888;
        }
        
        .metric-value {
            font-weight: 600;
            color: #fff;
        }
        
        .metric-value.highlight {
            color: #00ff88;
            font-size: 1.2em;
        }
        
        .progress-container {
            margin-top: 30px;
        }
        
        .progress-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
        }
        
        .progress-bar {
            height: 30px;
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            border-radius: 15px;
            transition: width 0.5s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            min-width: 60px;
        }
        
        .log-container {
            background: #0a0a15;
            border-radius: 10px;
            padding: 15px;
            max-height: 300px;
            overflow-y: auto;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.85em;
        }
        
        .log-line {
            padding: 3px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        
        .log-time {
            color: #666;
        }
        
        .log-content {
            color: #00ff88;
        }
        
        .chart-container {
            height: 200px;
            display: flex;
            align-items: flex-end;
            gap: 5px;
            padding: 20px 0;
        }
        
        .chart-bar {
            flex: 1;
            background: linear-gradient(180deg, #00d9ff, #0066ff);
            border-radius: 5px 5px 0 0;
            min-height: 10px;
            transition: height 0.3s ease;
        }
        
        .eta-card {
            text-align: center;
            padding: 30px;
        }
        
        .eta-time {
            font-size: 3em;
            font-weight: bold;
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .eta-label {
            color: #888;
            margin-top: 10px;
        }
        
        .refresh-note {
            text-align: center;
            color: #666;
            padding: 20px;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 AGI训练监控面板</h1>
            <div class="time" id="current-time"></div>
        </div>
        
        <div class="status-bar">
            <div class="status-item">
                <div class="status-dot" id="status-dot"></div>
                <span id="status-text">检查中...</span>
            </div>
            <div class="status-item">
                <span>PID: </span>
                <span id="pid">-</span>
            </div>
            <div class="status-item">
                <span>运行时间: </span>
                <span id="elapsed">-</span>
            </div>
        </div>
        
        <div class="grid">
            <div class="card">
                <h2>📊 训练指标</h2>
                <div class="metric">
                    <span class="metric-label">当前Epoch</span>
                    <span class="metric-value" id="epoch">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">训练Loss</span>
                    <span class="metric-value" id="train-loss">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">训练准确率</span>
                    <span class="metric-value highlight" id="train-acc">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">验证准确率</span>
                    <span class="metric-value highlight" id="val-acc">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">最佳准确率</span>
                    <span class="metric-value highlight" id="best-acc">-</span>
                </div>
            </div>
            
            <div class="card">
                <h2>⚡ 性能统计</h2>
                <div class="metric">
                    <span class="metric-label">处理速度</span>
                    <span class="metric-value" id="speed">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">总样本数</span>
                    <span class="metric-value" id="total-samples">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">模型参数</span>
                    <span class="metric-value" id="params">7,353,988</span>
                </div>
                <div class="metric">
                    <span class="metric-label">设备</span>
                    <span class="metric-value" id="device">MPS (Apple Silicon)</span>
                </div>
            </div>
            
            <div class="card eta-card">
                <h2>⏱️ 预计完成</h2>
                <div class="eta-time" id="eta">--:--:--</div>
                <div class="eta-label">剩余时间: <span id="remaining">-</span></div>
            </div>
        </div>
        
        <div class="card">
            <h2>📈 训练进度</h2>
            <div class="progress-container">
                <div class="progress-header">
                    <span>进度</span>
                    <span id="progress-text">0%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="progress-fill" style="width: 0%">0%</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>📝 实时日志</h2>
            <div class="log-container" id="log-container">
                <div class="log-line">等待日志...</div>
            </div>
        </div>
        
        <div class="refresh-note">
            页面每3秒自动刷新 | 
            <a href="#" onclick="location.reload()" style="color: #00d9ff;">手动刷新</a>
        </div>
    </div>
    
    <script>
        function updateTime() {
            const now = new Date();
            document.getElementById('current-time').textContent = 
                now.toLocaleString('zh-CN', { hour12: false });
        }
        
        async function fetchData() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                // 更新状态
                const statusDot = document.getElementById('status-dot');
                const statusText = document.getElementById('status-text');
                
                if (data.running) {
                    statusDot.className = 'status-dot running';
                    statusText.textContent = '训练运行中';
                } else {
                    statusDot.className = 'status-dot stopped';
                    statusText.textContent = '训练已停止';
                }
                
                document.getElementById('pid').textContent = data.pid || '-';
                document.getElementById('elapsed').textContent = data.elapsed || '-';
                
                // 更新指标
                document.getElementById('epoch').textContent = data.epoch || '-';
                document.getElementById('train-loss').textContent = data.train_loss || '-';
                document.getElementById('train-acc').textContent = data.train_acc || '-';
                document.getElementById('val-acc').textContent = data.val_acc || '-';
                document.getElementById('best-acc').textContent = data.best_acc || '-';
                document.getElementById('speed').textContent = data.speed || '-';
                document.getElementById('total-samples').textContent = data.total_samples || '-';
                
                // 更新进度
                const progress = parseFloat(data.progress) || 0;
                document.getElementById('progress-text').textContent = progress.toFixed(1) + '%';
                document.getElementById('progress-fill').style.width = progress + '%';
                document.getElementById('progress-fill').textContent = progress.toFixed(1) + '%';
                
                // 更新ETA
                document.getElementById('eta').textContent = data.eta || '--:--:--';
                document.getElementById('remaining').textContent = data.remaining || '-';
                
                // 更新日志
                if (data.logs && data.logs.length > 0) {
                    const logContainer = document.getElementById('log-container');
                    logContainer.innerHTML = data.logs.map(log => 
                        `<div class="log-line"><span class="log-content">${log}</span></div>`
                    ).join('');
                    logContainer.scrollTop = logContainer.scrollHeight;
                }
                
            } catch (error) {
                console.error('获取数据失败:', error);
            }
        }
        
        // 初始化
        updateTime();
        fetchData();
        
        // 定时更新
        setInterval(updateTime, 1000);
        setInterval(fetchData, 3000);
    </script>
</body>
</html>
'''


# ============================================================
# API端点
# ============================================================

def get_process_info() -> Dict:
    """获取进程信息"""
    result = subprocess.run(
        ['pgrep', '-f', 'optimized_5h_training.py'],
        capture_output=True, text=True
    )
    
    if result.returncode == 0:
        pid = result.stdout.strip().split('\n')[0]
        ps = subprocess.run(
            ['ps', '-p', pid, '-o', 'etime='],
            capture_output=True, text=True
        )
        return {
            'running': True,
            'pid': pid,
            'elapsed': ps.stdout.strip() if ps.returncode == 0 else '-'
        }
    return {'running': False, 'pid': None, 'elapsed': '-'}


def parse_log() -> Dict:
    """解析训练日志"""
    if not LOG_FILE.exists():
        return {}
    
    with open(LOG_FILE, 'r') as f:
        lines = f.readlines()
    
    result = {
        'epoch': '-',
        'train_loss': '-',
        'train_acc': '-',
        'val_acc': '-',
        'best_acc': '-',
        'progress': '0',
        'speed': '-',
        'eta': '--:--:--',
        'remaining': '-',
        'total_samples': '-',
        'logs': []
    }
    
    # 获取最近的日志行
    recent_logs = []
    for line in lines[-30:]:
        if 'Batch' in line or 'Epoch' in line or '完成' in line:
            # 清理日志行
            clean_line = line.strip()
            if '|' in clean_line:
                parts = clean_line.split('|')
                if len(parts) >= 2:
                    recent_logs.append(' | '.join(parts[1:]).strip())
    
    result['logs'] = recent_logs[-15:]
    
    # 解析Epoch完成信息
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        
        if 'Epoch' in line and '完成' in line:
            result['epoch'] = line.split()[1] if len(line.split()) > 1 else '-'
        
        if '训练 Loss:' in line:
            parts = line.split('|')
            for p in parts:
                if 'Loss:' in p:
                    result['train_loss'] = p.split(':')[1].strip()
                if 'Acc:' in p:
                    result['train_acc'] = p.split(':')[1].strip()
        
        if '验证 Acc:' in line:
            parts = line.split('|')
            for p in parts:
                if '验证 Acc:' in p:
                    result['val_acc'] = p.split(':')[1].strip()
                if '最佳:' in p:
                    result['best_acc'] = p.split(':')[1].strip()
        
        if '速度:' in line:
            parts = line.split('|')
            for p in parts:
                if '速度:' in p:
                    result['speed'] = p.split(':')[1].strip()
        
        if '进度:' in line:
            parts = line.split('|')
            for p in parts:
                if '进度:' in p:
                    progress_str = p.split(':')[1].strip().replace('%', '')
                    result['progress'] = progress_str
        
        if '预计完成:' in line:
            parts = line.split('预计完成:')
            if len(parts) > 1:
                result['eta'] = parts[1].strip()
        
        # 找到足够信息后退出
        if result['epoch'] != '-' and result['train_loss'] != '-':
            break
    
    # 计算剩余时间
    try:
        progress = float(result['progress'])
        if progress > 0:
            total_hours = 5.0
            elapsed_hours = total_hours * progress / 100
            remaining_hours = total_hours - elapsed_hours
            result['remaining'] = f"{remaining_hours:.1f} 小时"
    except:
        pass
    
    return result


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/status')
def api_status():
    proc_info = get_process_info()
    log_info = parse_log()
    
    return jsonify({
        **proc_info,
        **log_info
    })


# ============================================================
# 主函数
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("   🤖 AGI训练监控Web面板")
    print("=" * 60)
    print(f"   启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("   ")
    print("   🌐 请在浏览器中打开:")
    print("   ")
    print("      http://localhost:8080")
    print("   ")
    print("   按 Ctrl+C 停止监控服务")
    print("=" * 60 + "\n")
    
    app.run(host='0.0.0.0', port=8080, debug=False)


if __name__ == "__main__":
    main()
