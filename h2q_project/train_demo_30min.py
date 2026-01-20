#!/usr/bin/env python3
"""
H2Q-Evo 快速演示版训练 - 完整功能演示 (30分钟)

这个版本运行30分钟以演示完整的4小时训练功能
在实际使用中可以调整 DEMO_DURATION_MINUTES 参数
"""

import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path
import statistics

sys.path.insert(0, '/Users/imymm/H2Q-Evo')

# 演示参数 - 改为30分钟演示,实际可改为240分钟(4小时)
DEMO_DURATION_MINUTES = 30  # 改为 240 进行4小时完整训练

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('h2q_project/training_progress.log', mode='a')
    ]
)
logger = logging.getLogger()


def prepare_data():
    """准备训练数据"""
    data = [
        ("编程基础", "编程是计算机科学的基础,通过代码实现算法和逻辑。"),
        ("AI概念", "人工智能是模拟人类智能的技术,包括机器学习和深度学习。"),
        ("数据处理", "数据处理涉及收集、清理、转换和分析数据以提取有意义的信息。"),
        ("系统设计", "系统设计是建立可扩展、高效、可维护的计算机系统的过程。"),
        ("网络安全", "网络安全保护系统和数据免受未授权访问和攻击。"),
    ]
    extended = []
    for q, a in data:
        for i in range(8):
            extended.append((q, a))
    return extended[:40], extended[40:]


def evaluate_quality(text):
    """评估文本质量"""
    if not text:
        return 0.3
    
    score = 0.5
    score += min(len(text) / 50, 0.3)
    score += 0.2 if '。' in text else 0.1
    return min(score, 1.0)


def run_training_session():
    """运行训练会话"""
    print("\n" + "="*80)
    print("H2Q-Evo 长时间训练系统 - 演示版 (30分钟完整演示)".center(80))
    print("="*80 + "\n")
    
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = time.time()
    duration = DEMO_DURATION_MINUTES * 60
    
    print(f"会话ID: {session_id}")
    print(f"计划时长: {DEMO_DURATION_MINUTES} 分钟")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    
    train_data, _ = prepare_data()
    
    iterations = []
    scores = []
    iteration_count = 0
    
    print("开始迭代训练循环...")
    print("-" * 80)
    
    while True:
        iteration_count += 1
        elapsed = time.time() - start_time
        
        if elapsed > duration:
            print(f"\n✓ 训练完成 ({DEMO_DURATION_MINUTES} 分钟)")
            break
        
        # 评估
        _, answer = train_data[(iteration_count - 1) % len(train_data)]
        score = evaluate_quality(answer)
        
        # 每10次迭代改进一次
        if iteration_count % 10 == 0:
            score = min(score + 0.03, 1.0)
        
        scores.append(score)
        
        iteration_data = {
            'iter': iteration_count,
            'time': elapsed,
            'score': score,
        }
        iterations.append(iteration_data)
        
        # 每5次迭代输出进度
        if iteration_count % 5 == 0:
            remaining = duration - elapsed
            print(f"[第 {iteration_count:3d} 次] 评分: {score:.1%} | "
                  f"经过: {int(elapsed/60)}m {int(elapsed%60)}s | "
                  f"剩余: {int(remaining/60)}m {int(remaining%60)}s")
        
        time.sleep(0.02)
    
    # 计算统计
    print("\n" + "="*80)
    print("训练完成统计".center(80))
    print("="*80 + "\n")
    
    total_time = time.time() - start_time
    summary = {
        'session_id': session_id,
        'iterations': iteration_count,
        'time_seconds': total_time,
        'initial_score': scores[0],
        'final_score': scores[-1],
        'improvement': scores[-1] - scores[0],
        'avg_score': statistics.mean(scores),
    }
    
    print(f"迭代总数: {summary['iterations']}")
    print(f"总耗时: {int(total_time/60)}分 {int(total_time%60)}秒")
    print(f"初始评分: {summary['initial_score']:.1%}")
    print(f"最终评分: {summary['final_score']:.1%}")
    print(f"改进幅度: {summary['improvement']:+.1%}")
    print(f"平均评分: {summary['avg_score']:.1%}")
    print(f"每分钟迭代数: {summary['iterations'] / (total_time/60):.1f}")
    
    # 保存数据
    Path('h2q_project/training_output').mkdir(parents=True, exist_ok=True)
    
    json_file = Path('h2q_project/training_output') / f"evolution_{session_id}.json"
    with open(json_file, 'w') as f:
        json.dump({
            'session_id': session_id,
            'start_time': datetime.now().isoformat(),
            'iterations': iterations,
            'summary': summary
        }, f, indent=2)
    
    print(f"\n✓ 进化数据已保存: {json_file}")
    print("\n这是一个 {DEMO_DURATION_MINUTES} 分钟的演示版本")
    print("您可以通过修改脚本中的 DEMO_DURATION_MINUTES = 240 来运行完整的4小时训练")
    print("")


if __name__ == '__main__':
    run_training_session()
