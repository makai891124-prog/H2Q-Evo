#!/usr/bin/env python3
"""
AGI训练效果分析报告生成器
"""
import json
import os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

def generate_training_analysis_report():
    """生成训练效果分析报告"""

    # 读取训练报告
    report_path = Path("reports/training_report.json")
    if not report_path.exists():
        print("❌ 找不到训练报告文件")
        return None

    with open(report_path, 'r', encoding='utf-8') as f:
        report_data = json.load(f)

    # 分析训练效果
    train_losses = report_data['training_history']['train_losses']
    val_losses = report_data['training_history']['val_losses']

    analysis = {
        'summary': {
            'model_type': report_data['training_summary']['model_type'],
            'total_epochs': report_data['training_summary']['total_epochs'],
            'final_train_loss': report_data['training_summary']['final_train_loss'],
            'final_val_loss': report_data['training_summary']['final_val_loss'],
            'best_val_loss': report_data['training_summary']['best_val_loss'],
            'training_timestamp': report_data['timestamp'],
            'algorithm_used': report_data['algorithm_used']
        },
        'performance_metrics': {
            'convergence_rate': calculate_convergence_rate(train_losses),
            'stability_score': calculate_stability_score(val_losses),
            'improvement_ratio': calculate_improvement_ratio(train_losses),
            'overfitting_indicator': calculate_overfitting_indicator(train_losses, val_losses)
        },
        'training_characteristics': {
            'loss_reduction_pattern': analyze_loss_pattern(train_losses),
            'validation_trend': analyze_validation_trend(val_losses),
            'learning_efficiency': analyze_learning_efficiency(train_losses, val_losses)
        },
        'recommendations': generate_training_recommendations(train_losses, val_losses)
    }

    # 保存详细分析报告
    analysis_path = Path("reports/training_analysis_report.json")
    analysis_path.parent.mkdir(exist_ok=True)

    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    print(f"✅ 训练分析报告已保存到: {analysis_path}")
    return analysis

def calculate_convergence_rate(losses):
    """计算收敛速率"""
    if len(losses) < 2:
        return 0.0

    initial_loss = losses[0]
    final_loss = losses[-1]
    total_epochs = len(losses)

    # 计算损失减少的速率
    convergence_rate = (initial_loss - final_loss) / (initial_loss * total_epochs)
    return max(0.0, convergence_rate)

def calculate_stability_score(losses):
    """计算稳定性分数"""
    if len(losses) < 3:
        return 0.5

    # 计算损失的标准差（越小越稳定）
    std_dev = np.std(losses)
    mean_loss = np.mean(losses)

    # 标准化稳定性分数 (0-1, 1表示非常稳定)
    if mean_loss == 0:
        return 1.0

    stability_score = 1.0 / (1.0 + (std_dev / mean_loss))
    return stability_score

def calculate_improvement_ratio(losses):
    """计算改进比率"""
    if len(losses) < 2:
        return 0.0

    # 计算前50%和后50%的平均损失对比
    midpoint = len(losses) // 2
    early_avg = np.mean(losses[:midpoint])
    late_avg = np.mean(losses[midpoint:])

    if early_avg == 0:
        return 1.0

    improvement_ratio = (early_avg - late_avg) / early_avg
    return max(0.0, improvement_ratio)

def calculate_overfitting_indicator(train_losses, val_losses):
    """计算过拟合指标"""
    if len(train_losses) != len(val_losses):
        return 0.5

    # 计算训练损失和验证损失的差异趋势
    train_final = train_losses[-1]
    val_final = val_losses[-1]

    if train_final == 0:
        return 0.0

    overfitting_ratio = (val_final - train_final) / train_final
    return max(0.0, min(1.0, overfitting_ratio))

def analyze_loss_pattern(losses):
    """分析损失模式"""
    if len(losses) < 3:
        return "数据不足"

    # 检查是否单调递减
    decreasing = all(losses[i] >= losses[i+1] for i in range(len(losses)-1))

    # 计算损失变化的平滑度
    diffs = [abs(losses[i+1] - losses[i]) for i in range(len(losses)-1)]
    avg_change = np.mean(diffs)
    max_change = max(diffs)

    if decreasing and avg_change < 0.01:
        return "平滑收敛"
    elif decreasing:
        return "稳步下降"
    else:
        return "波动较大"

def analyze_validation_trend(val_losses):
    """分析验证趋势"""
    if len(val_losses) < 3:
        return "数据不足"

    # 检查验证损失是否持续改善
    improving = val_losses[0] > val_losses[-1]

    # 计算验证损失的稳定性
    stability = calculate_stability_score(val_losses)

    if improving and stability > 0.8:
        return "稳定改善"
    elif improving:
        return "逐步改善"
    else:
        return "需要调整"

def analyze_learning_efficiency(train_losses, val_losses):
    """分析学习效率"""
    convergence = calculate_convergence_rate(train_losses)
    stability = calculate_stability_score(val_losses)

    efficiency_score = (convergence + stability) / 2

    if efficiency_score > 0.8:
        return "高效学习"
    elif efficiency_score > 0.6:
        return "良好学习"
    elif efficiency_score > 0.4:
        return "一般学习"
    else:
        return "学习效率待改善"

def generate_training_recommendations(train_losses, val_losses):
    """生成训练建议"""
    recommendations = []

    convergence_rate = calculate_convergence_rate(train_losses)
    stability_score = calculate_stability_score(val_losses)
    overfitting = calculate_overfitting_indicator(train_losses, val_losses)

    if convergence_rate < 0.1:
        recommendations.append("考虑增加学习率或调整优化器")
        recommendations.append("检查数据质量和预处理")

    if stability_score < 0.5:
        recommendations.append("增加正则化技术（如dropout, weight decay）")
        recommendations.append("尝试更小的批次大小")

    if overfitting > 0.3:
        recommendations.append("实施早停机制")
        recommendations.append("增加数据增强或正则化")

    if len(recommendations) == 0:
        recommendations.append("训练效果良好，可以考虑扩展数据集")
        recommendations.append("尝试更复杂的模型架构")

    return recommendations

def create_visualization_report():
    """创建可视化报告"""
    try:
        # 读取训练数据
        report_path = Path("reports/training_report.json")
        if not report_path.exists():
            return

        with open(report_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        train_losses = data['training_history']['train_losses']
        val_losses = data['training_history']['val_losses']
        epochs = list(range(1, len(train_losses) + 1))

        # 创建图表
        plt.figure(figsize=(12, 8))

        # 损失曲线
        plt.subplot(2, 2, 1)
        plt.plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2)
        plt.plot(epochs, val_losses, 'r-', label='验证损失', linewidth=2)
        plt.xlabel('训练轮次')
        plt.ylabel('损失值')
        plt.title('训练和验证损失曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 损失差值
        plt.subplot(2, 2, 2)
        loss_diff = [v - t for t, v in zip(train_losses, val_losses)]
        plt.plot(epochs, loss_diff, 'g-', linewidth=2)
        plt.xlabel('训练轮次')
        plt.ylabel('验证损失 - 训练损失')
        plt.title('过拟合指标')
        plt.grid(True, alpha=0.3)

        # 收敛分析
        plt.subplot(2, 2, 3)
        if len(train_losses) > 1:
            convergence = [(train_losses[0] - loss) / train_losses[0] for loss in train_losses]
            plt.plot(epochs, convergence, 'purple', linewidth=2)
            plt.xlabel('训练轮次')
            plt.ylabel('收敛程度')
            plt.title('训练收敛分析')
            plt.grid(True, alpha=0.3)

        # 稳定性分析
        plt.subplot(2, 2, 4)
        window_size = min(5, len(val_losses))
        if len(val_losses) >= window_size:
            stability = []
            for i in range(window_size, len(val_losses) + 1):
                window = val_losses[i-window_size:i]
                stability.append(1.0 / (1.0 + np.std(window)))
            plt.plot(range(window_size, len(val_losses) + 1), stability, 'orange', linewidth=2)
            plt.xlabel('训练轮次')
            plt.ylabel('稳定性分数')
            plt.title('训练稳定性分析')
            plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图表
        chart_path = Path("reports/training_analysis_chart.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ 训练分析图表已保存到: {chart_path}")

    except Exception as e:
        print(f"⚠️  创建可视化报告失败: {e}")

def main():
    """主函数"""
    print("📊 生成AGI训练效果分析报告")
    print("=" * 50)

    # 生成分析报告
    analysis = generate_training_analysis_report()

    if analysis:
        print("\n📈 训练效果分析结果:")
        print(f"   模型类型: {analysis['summary']['model_type']}")
        print(f"   总训练轮次: {analysis['summary']['total_epochs']}")
        print(f"   最终训练损失: {analysis['summary']['final_train_loss']:.4f}")
        print(f"   最终验证损失: {analysis['summary']['final_val_loss']:.4f}")
        print(f"   最佳验证损失: {analysis['summary']['best_val_loss']:.4f}")
        print("\n🎯 性能指标:")
        print(f"   收敛速率: {analysis['performance_metrics']['convergence_rate']:.4f}")
        print(f"   稳定性分数: {analysis['performance_metrics']['stability_score']:.4f}")
        print(f"   改进比率: {analysis['performance_metrics']['improvement_ratio']:.4f}")
        print(f"   过拟合指标: {analysis['performance_metrics']['overfitting_indicator']:.4f}")
        print("\n🔍 训练特征:")
        print(f"   损失模式: {analysis['training_characteristics']['loss_reduction_pattern']}")
        print(f"   验证趋势: {analysis['training_characteristics']['validation_trend']}")
        print(f"   学习效率: {analysis['training_characteristics']['learning_efficiency']}")

        print("\n💡 训练建议:")
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"   {i}. {rec}")

    # 创建可视化报告
    create_visualization_report()

    print("\n" + "=" * 50)
    print("✅ 训练效果分析报告生成完成！")
    print("📁 查看 reports/ 目录获取详细报告和图表")

if __name__ == "__main__":
    main()