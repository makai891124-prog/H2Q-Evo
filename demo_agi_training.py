#!/usr/bin/env python3
"""
H2Q-Evo AGI训练系统演示
展示完整训练基础设施的运行
"""

import time
import json
from pathlib import Path

def demonstrate_infrastructure():
    """演示基础设施组件"""
    print("=== 演示AGI训练基础设施 ===")

    from agi_training_infrastructure import start_agi_infrastructure

    # 启动基础设施
    infra = start_agi_infrastructure()
    print("基础设施已启动")

    # 等待几秒让监控启动
    time.sleep(3)

    # 获取状态
    status = infra.get_system_status()
    print(f"系统状态: {json.dumps(status, indent=2, ensure_ascii=False)[:500]}...")

    # 停止基础设施
    infra.stop_infrastructure()
    print("基础设施已停止")

def demonstrate_checkpoint_system():
    """演示检查点系统"""
    print("\n=== 演示检查点系统 ===")

    from agi_checkpoint_system import create_training_state, save_model_checkpoint, load_model_checkpoint, get_checkpoint_manager

    # 创建训练状态
    training_state = create_training_state(
        model_weights={'layer1': [1, 2, 3], 'layer2': [4, 5, 6]},
        optimizer_state={'lr': 0.001},
        epoch=1,
        step=100
    )
    print("训练状态已创建")

    # 保存检查点
    version = save_model_checkpoint(training_state, 1, 0.85, 0.45)
    print(f"检查点已保存: {version}")

    # 加载检查点
    loaded_state = load_model_checkpoint(version)
    if loaded_state:
        print(f"检查点已加载: epoch={loaded_state.epoch}, step={loaded_state.step}")

    # 列出检查点
    manager = get_checkpoint_manager()
    checkpoints = manager.list_checkpoints()
    print(f"检查点数量: {len(checkpoints)}")

def demonstrate_fault_tolerance():
    """演示容错系统"""
    print("\n=== 演示容错系统 ===")

    from agi_fault_tolerance import get_fault_tolerance_manager, FaultType

    ft_manager = get_fault_tolerance_manager()

    # 注册健康检查
    def dummy_check():
        return True

    ft_manager.register_health_check("demo_check", dummy_check, 30)
    print("健康检查已注册")

    # 报告故障
    fault_id = ft_manager.report_fault(
        FaultType.NETWORK_ERROR,
        'low',
        "演示网络故障",
        {'demo': True}
    )
    print(f"故障已报告: {fault_id}")

    # 获取健康状态
    health = ft_manager.get_system_health()
    print(f"系统健康: {health['overall_health']}")

def demonstrate_realtime_training():
    """演示实时训练系统"""
    print("\n=== 演示实时训练系统 ===")

    from agi_realtime_training import H2QRealtimeTrainer, RealtimeTrainingConfig

    # 创建配置
    config = RealtimeTrainingConfig()
    config.training_enabled = False  # 演示模式，不实际训练
    config.hot_generation_enabled = False
    config.continuous_operation = False

    # 创建训练器
    trainer = H2QRealtimeTrainer(config)
    print("实时训练器已创建")

    # 获取状态
    status = trainer.get_training_status()
    print(f"训练状态: 运行={status['running']}, 步骤={status['current_step']}")

def create_demo_report():
    """创建演示报告"""
    print("\n=== 创建演示报告 ===")

    report = {
        'timestamp': time.time(),
        'components_tested': [
            'agi_training_infrastructure',
            'agi_checkpoint_system',
            'agi_fault_tolerance',
            'agi_realtime_training'
        ],
        'status': 'success',
        'notes': '所有组件演示成功'
    }

    with open('agi_demo_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("演示报告已创建: agi_demo_report.json")

def main():
    """主演示函数"""
    print("H2Q-Evo AGI训练系统演示")
    print("=" * 50)

    try:
        # 演示各个组件
        demonstrate_infrastructure()
        demonstrate_checkpoint_system()
        demonstrate_fault_tolerance()
        demonstrate_realtime_training()

        # 创建报告
        create_demo_report()

        print("\n" + "=" * 50)
        print("演示完成！所有组件运行正常")
        print("查看 agi_demo_report.json 获取详细报告")

    except Exception as e:
        print(f"演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()