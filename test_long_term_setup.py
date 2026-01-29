#!/usr/bin/env python3
"""
测试24-48小时长期运行设置
"""

import sys
import asyncio
import os
import json
import time
sys.path.append('.')

from true_agi_autonomous_system import TrueAGIAutonomousSystem, LearningExperience

async def test_long_term_setup():
    print('🚀 测试24-48小时长期运行设置...')

    # 初始化系统
    system = TrueAGIAutonomousSystem(256, 64)

    # 模拟运行几步
    system.is_running = True
    system.evolution_step = 995  # 接近1000步保存点

    for i in range(10):
        try:
            # 感知环境
            current_state, images, videos = system._perceive_environment()

            # 计算意识
            consciousness, internal_state = system.consciousness_engine(current_state, system.prev_consciousness_state)
            system.prev_consciousness_state = internal_state

            # 生成目标
            if len(system.goal_system.active_goals) < 1:
                system.goal_system.generate_goal(current_state, consciousness)

            # 选择动作
            action = system.learning_engine.select_action(current_state, images=images, videos=videos)

            # 执行动作
            reward, next_state = await system._execute_action(action)

            # 创建经验
            experience = LearningExperience(
                observation=current_state,
                action=action,
                reward=reward,
                next_observation=next_state,
                done=False,
                timestamp=time.time(),
                complexity=consciousness.neural_complexity
            )

            # 学习
            learning_metrics = system.learning_engine.learn_from_experience(experience, images=images, videos=videos)

            # 更新目标
            completed_goals = system.goal_system.update_goals(next_state, learning_metrics)

            # 记录状态
            system.performance_history.append(consciousness)
            system.learning_history.append(learning_metrics)

            # 更新状态
            system.current_state = next_state
            system.evolution_step += 1

            # 检查保存条件
            current_time = time.time()
            if (system.evolution_step % 1000 == 0 or
                current_time - getattr(system, 'last_save_time', 0) > 3600):
                print(f'📊 触发保存条件: 步数={system.evolution_step}, 时间差={current_time - system.last_save_time:.1f}秒')
                system.save_state('test_agi_system_state.json')
                system._save_monitoring_data()
                system.last_save_time = current_time

            print(f'✅ 步骤 {system.evolution_step} 完成')

        except Exception as e:
            print(f'❌ 步骤 {i} 出错: {e}')
            break

    # 检查输出文件
    if os.path.exists('agi_monitoring_data.jsonl'):
        print('✅ 监控数据文件已创建')
        with open('agi_monitoring_data.jsonl', 'r') as f:
            lines = f.readlines()
            print(f'📊 监控数据行数: {len(lines)}')
            if lines:
                data = json.loads(lines[0])
                print(f'📈 示例监控数据: evolution_step={data.get("evolution_step")}, knowledge_base_size={data.get("knowledge_base_size")}')
    else:
        print('❌ 监控数据文件未创建')

    if os.path.exists('test_agi_system_state.json'):
        print('✅ 系统状态文件已创建')
    else:
        print('❌ 系统状态文件未创建')

    print('🎯 长期运行设置测试完成')

if __name__ == "__main__":
    asyncio.run(test_long_term_setup())