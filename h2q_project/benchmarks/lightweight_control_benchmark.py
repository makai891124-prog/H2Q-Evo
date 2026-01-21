"""H2Q Lightweight Control Core - 综合基准测试.

测试内容:
1. 模型大小验证 (目标: <2KB)
2. 确定性验证 (相同输入 → 相同输出)
3. 微变化检测灵敏度
4. 轨迹预测准确率
5. 实时性能 (延迟 <50μs)
6. 相位跃迁检测
7. 长时间稳定性
8. 噪声鲁棒性
"""

import numpy as np
import time
import json
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from h2q.control.lightweight_control import (
    create_lightweight_control,
    H2QLightweightControl,
)


@dataclass
class BenchmarkResult:
    """基准测试结果."""
    test_name: str
    passed: bool
    score: float
    details: Dict[str, Any]


class LightweightBenchmark:
    """轻量级控制核心基准测试套件."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        
    def run_all(self) -> Dict[str, Any]:
        """运行所有基准测试."""
        print("="*70)
        print("H2Q Lightweight Control Core - 综合基准测试")
        print("="*70)
        
        tests = [
            self.test_model_size,
            self.test_determinism,
            self.test_micro_change_detection,
            self.test_trajectory_prediction,
            self.test_latency_performance,
            self.test_phase_transition_detection,
            self.test_long_running_stability,
            self.test_noise_robustness,
            self.test_real_world_scenario,
        ]
        
        for test in tests:
            try:
                result = test()
                self.results.append(result)
                status = "✅ PASS" if result.passed else "❌ FAIL"
                print(f"\n{status} {result.test_name}")
                print(f"   Score: {result.score:.2f}")
                for key, value in result.details.items():
                    if isinstance(value, float):
                        print(f"   {key}: {value:.4f}")
                    elif isinstance(value, list) and len(value) <= 10:
                        formatted = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in value]
                        print(f"   {key}: {formatted}")
                    else:
                        print(f"   {key}: {value}")
            except Exception as e:
                import traceback
                print(f"\n❌ ERROR {test.__name__}: {e}")
                traceback.print_exc()
                self.results.append(BenchmarkResult(
                    test_name=test.__name__,
                    passed=False,
                    score=0.0,
                    details={'error': str(e)}
                ))
        
        return self.generate_report()
    
    def test_model_size(self) -> BenchmarkResult:
        """测试 1: 模型大小验证."""
        controller = create_lightweight_control(input_dim=16)
        
        size_bytes = controller.model_size_bytes()
        param_count = controller.parameter_count()
        size_kb = size_bytes / 1024
        
        # 目标: <2KB
        target_kb = 2.0
        passed = size_kb < target_kb
        score = min(1.0, target_kb / size_kb) if size_kb > 0 else 1.0
        
        return BenchmarkResult(
            test_name="Model Size (目标 <2KB)",
            passed=passed,
            score=score,
            details={
                'parameters': param_count,
                'size_bytes': size_bytes,
                'size_kb': size_kb,
                'target_kb': target_kb,
            }
        )
    
    def test_determinism(self) -> BenchmarkResult:
        """测试 2: 确定性验证."""
        # 创建两个相同配置的控制器
        c1 = create_lightweight_control(input_dim=8, seed=42)
        c2 = create_lightweight_control(input_dim=8, seed=42)
        
        # 生成固定输入
        np.random.seed(123)
        inputs = [np.random.randn(8) for _ in range(100)]
        
        all_identical = True
        max_diff = 0.0
        
        for signal in inputs[:50]:
            # 重置后处理相同信号
            c1.reset()
            c2.reset()
            
            s1 = c1.process(signal.copy())
            s2 = c2.process(signal.copy())
            
            diff = np.abs(s1.quaternion - s2.quaternion).max()
            max_diff = max(max_diff, diff)
            
            if diff > 1e-10:
                all_identical = False
        
        passed = all_identical
        score = 1.0 if passed else max(0.0, 1.0 - max_diff)
        
        return BenchmarkResult(
            test_name="Determinism (确定性输出)",
            passed=passed,
            score=score,
            details={
                'identical_outputs': all_identical,
                'max_difference': max_diff,
                'test_samples': 50,
            }
        )
    
    def test_micro_change_detection(self) -> BenchmarkResult:
        """测试 3: 微变化检测灵敏度."""
        controller = create_lightweight_control(
            input_dim=8,
            anomaly_sensitivity=0.1,  # 高灵敏度
        )
        
        np.random.seed(42)
        
        # 校准阶段
        baseline = np.zeros(8)
        for _ in range(100):
            noise = np.random.randn(8) * 0.01
            controller.process(baseline + noise)
        
        # 测试不同幅度的扰动
        perturbation_levels = [0.01, 0.05, 0.1, 0.5, 1.0]
        detection_results = []
        
        for level in perturbation_levels:
            detections = 0
            trials = 50
            
            for _ in range(trials):
                # 先发几个正常信号
                for _ in range(5):
                    controller.process(baseline + np.random.randn(8) * 0.01)
                
                # 注入扰动
                perturbed = baseline.copy()
                perturbed[0] += level
                state = controller.process(perturbed)
                
                if state.anomaly_score > 0.5 or state.is_anomaly:
                    detections += 1
            
            detection_results.append(detections / trials)
        
        # 评分: 检测率应该随扰动增加而提高
        avg_detection = np.mean(detection_results)
        
        # 检查是否大扰动有更高检测率
        large_perturbation_rate = np.mean(detection_results[-2:])
        
        passed = large_perturbation_rate > 0.5
        score = large_perturbation_rate
        
        return BenchmarkResult(
            test_name="Micro-Change Detection (微变化检测)",
            passed=passed,
            score=score,
            details={
                'perturbation_levels': perturbation_levels,
                'detection_rates': detection_results,
                'avg_detection_rate': avg_detection,
                'large_perturbation_rate': large_perturbation_rate,
            }
        )
    
    def test_trajectory_prediction(self) -> BenchmarkResult:
        """测试 4: 轨迹预测准确率."""
        controller = create_lightweight_control(
            input_dim=4,
            prediction_horizon=10,
        )
        
        # 生成平滑轨迹 (正弦波)
        t = np.linspace(0, 4*np.pi, 200)
        trajectory = np.column_stack([
            np.sin(t),
            np.cos(t),
            np.sin(2*t) * 0.5,
            np.cos(2*t) * 0.5,
        ])
        
        # 处理轨迹并评估预测
        phase_predictions_correct = 0
        total_predictions = 0
        
        for i in range(50, len(trajectory) - 20):
            signal = trajectory[i]
            state = controller.process(signal)
            
            # 获取预测
            pred = controller.predict_trajectory()
            
            if pred is not None:
                total_predictions += 1
                
                # 计算预测的相位变化方向
                curr_phase = state.phase
                
                # 比较预测趋势与实际趋势
                actual_future = trajectory[i+1:i+6]
                actual_phases = []
                for af in actual_future:
                    norm = np.sqrt(np.sum(af * af))
                    if norm > 1e-8:
                        af_norm = af / norm
                        w = af_norm[0]
                        v_norm = np.sqrt(af_norm[1]**2 + af_norm[2]**2 + af_norm[3]**2)
                        actual_phases.append(2 * np.arctan2(v_norm, w))
                
                if len(actual_phases) >= 2:
                    actual_trend = np.sign(actual_phases[-1] - actual_phases[0])
                    
                    pred_phases = []
                    for p in pred[:5]:
                        w = p[0]
                        v_norm = np.sqrt(p[1]**2 + p[2]**2 + p[3]**2)
                        pred_phases.append(2 * np.arctan2(v_norm, w))
                    
                    if len(pred_phases) >= 2:
                        pred_trend = np.sign(pred_phases[-1] - pred_phases[0])
                        
                        if pred_trend == actual_trend or actual_trend == 0:
                            phase_predictions_correct += 1
        
        accuracy = phase_predictions_correct / total_predictions if total_predictions > 0 else 0
        
        passed = accuracy > 0.4  # 趋势预测准确率 >40%
        score = accuracy
        
        return BenchmarkResult(
            test_name="Trajectory Prediction (轨迹趋势预测)",
            passed=passed,
            score=score,
            details={
                'total_predictions': total_predictions,
                'correct_trend_predictions': phase_predictions_correct,
                'trend_accuracy': accuracy,
            }
        )
    
    def test_latency_performance(self) -> BenchmarkResult:
        """测试 5: 实时性能."""
        controller = create_lightweight_control(input_dim=16)
        
        np.random.seed(42)
        
        # 预热
        for _ in range(100):
            controller.process(np.random.randn(16))
        
        controller._latencies.clear()
        
        # 计时测试
        latencies = []
        for _ in range(10000):
            signal = np.random.randn(16)
            
            start = time.perf_counter()
            controller.process(signal)
            end = time.perf_counter()
            
            latencies.append((end - start) * 1e6)
        
        mean_latency = np.mean(latencies)
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        max_latency = np.max(latencies)
        min_latency = np.min(latencies)
        
        # 目标: 平均延迟 <50μs
        target_us = 50.0
        passed = mean_latency < target_us
        score = min(1.0, target_us / mean_latency) if mean_latency > 0 else 1.0
        
        return BenchmarkResult(
            test_name="Latency Performance (延迟 <50μs)",
            passed=passed,
            score=score,
            details={
                'mean_latency_us': mean_latency,
                'min_latency_us': min_latency,
                'p50_latency_us': p50_latency,
                'p95_latency_us': p95_latency,
                'p99_latency_us': p99_latency,
                'max_latency_us': max_latency,
                'target_us': target_us,
                'samples': len(latencies),
            }
        )
    
    def test_phase_transition_detection(self) -> BenchmarkResult:
        """测试 6: 相位跃迁检测."""
        controller = create_lightweight_control(input_dim=4, history_len=32)
        
        np.random.seed(42)
        
        # 测试多个跃迁场景
        transitions_injected = 0
        transitions_detected = 0
        
        for trial in range(20):
            controller.reset()
            
            # 稳定阶段
            for i in range(30):
                signal = np.array([0.1, 0.1, 0.1, 0.1]) + np.random.randn(4) * 0.01
                controller.process(signal)
            
            # 注入跃迁 (突然的大幅度变化)
            transition_detected_in_trial = False
            transitions_injected += 1
            
            for i in range(10):
                if i == 5:
                    # 突变
                    signal = np.array([1.0, -1.0, 1.0, -1.0])
                else:
                    signal = np.array([0.1, 0.1, 0.1, 0.1]) + np.random.randn(4) * 0.01
                
                state = controller.process(signal)
                
                if state.phase_transition:
                    transition_detected_in_trial = True
            
            if transition_detected_in_trial:
                transitions_detected += 1
        
        detection_rate = transitions_detected / transitions_injected if transitions_injected > 0 else 0
        
        passed = detection_rate >= 0.5
        score = detection_rate
        
        return BenchmarkResult(
            test_name="Phase Transition Detection (相位跃迁检测)",
            passed=passed,
            score=score,
            details={
                'injected_transitions': transitions_injected,
                'detected_transitions': transitions_detected,
                'detection_rate': detection_rate,
            }
        )
    
    def test_long_running_stability(self) -> BenchmarkResult:
        """测试 7: 长时间运行稳定性."""
        controller = create_lightweight_control(input_dim=8)
        
        np.random.seed(42)
        
        num_steps = 1000000  # 100万步
        
        start_time = time.time()
        
        numerical_stable = True
        
        for i in range(num_steps):
            signal = np.random.randn(8) * 0.1
            
            # 偶尔注入异常
            if i % 100000 == 50000:
                signal += np.random.randn(8) * 2.0
            
            state = controller.process(signal)
            
            # 检查数值稳定性
            if np.isnan(state.quaternion).any() or np.isinf(state.quaternion).any():
                numerical_stable = False
                break
        
        end_time = time.time()
        elapsed = end_time - start_time
        throughput = num_steps / elapsed
        
        passed = numerical_stable
        score = 1.0 if passed else 0.0
        
        return BenchmarkResult(
            test_name="Long-Running Stability (长时间稳定性)",
            passed=passed,
            score=score,
            details={
                'total_steps': num_steps,
                'elapsed_seconds': elapsed,
                'throughput_steps_per_sec': throughput,
                'numerical_stable': numerical_stable,
                'final_samples': controller.metrics.total_samples,
                'mean_latency_us': controller.metrics.mean_latency_us,
            }
        )
    
    def test_noise_robustness(self) -> BenchmarkResult:
        """测试 8: 噪声鲁棒性."""
        np.random.seed(42)
        
        noise_levels = [0.001, 0.01, 0.05, 0.1]
        false_positive_rates = []
        
        for noise_std in noise_levels:
            controller = create_lightweight_control(
                input_dim=8,
                anomaly_sensitivity=0.05,
            )
            
            base_signal = np.array([0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005])
            
            # 校准
            for _ in range(100):
                noisy = base_signal + np.random.randn(8) * noise_std
                controller.process(noisy)
            
            # 测试
            false_positives = 0
            trials = 500
            
            for _ in range(trials):
                noisy = base_signal + np.random.randn(8) * noise_std
                state = controller.process(noisy)
                
                # 在正常噪声下不应该频繁触发异常
                if state.is_anomaly:
                    false_positives += 1
            
            fpr = false_positives / trials
            false_positive_rates.append(fpr)
        
        # 低噪声时假阳性率应该低
        low_noise_fpr = np.mean(false_positive_rates[:2])
        
        passed = low_noise_fpr < 0.05
        score = 1.0 - low_noise_fpr
        
        return BenchmarkResult(
            test_name="Noise Robustness (噪声鲁棒性)",
            passed=passed,
            score=score,
            details={
                'noise_levels': noise_levels,
                'false_positive_rates': false_positive_rates,
                'low_noise_fpr': low_noise_fpr,
            }
        )
    
    def test_real_world_scenario(self) -> BenchmarkResult:
        """测试 9: 真实场景模拟 (工业传感器)."""
        controller = create_lightweight_control(
            input_dim=6,  # 6轴传感器
            anomaly_sensitivity=0.05,
        )
        
        np.random.seed(42)
        
        # 模拟工业传感器数据
        # 正常运行: 低幅度振动
        # 异常: 突然的高幅度尖峰
        
        correct_detections = 0
        false_alarms = 0
        missed_detections = 0
        
        # 校准
        for _ in range(200):
            signal = np.random.randn(6) * 0.05  # 正常振动
            controller.process(signal)
        
        # 运行测试
        is_anomaly_period = False
        anomaly_counter = 0
        
        for i in range(1000):
            # 每100步有10%概率进入异常期
            if i % 100 == 0:
                is_anomaly_period = np.random.random() < 0.1
                anomaly_counter = 0
            
            if is_anomaly_period and anomaly_counter < 5:
                # 异常信号 (高幅度尖峰)
                signal = np.random.randn(6) * 0.5 + np.array([1.0, -0.5, 0.8, -0.3, 0.6, -0.4])
                anomaly_counter += 1
                is_real_anomaly = True
            else:
                # 正常信号
                signal = np.random.randn(6) * 0.05
                is_real_anomaly = False
            
            state = controller.process(signal)
            detected = state.is_anomaly or state.anomaly_score > 0.5
            
            if is_real_anomaly and detected:
                correct_detections += 1
            elif is_real_anomaly and not detected:
                missed_detections += 1
            elif not is_real_anomaly and detected:
                false_alarms += 1
        
        total_real_anomalies = correct_detections + missed_detections
        
        precision = correct_detections / (correct_detections + false_alarms) if (correct_detections + false_alarms) > 0 else 0
        recall = correct_detections / total_real_anomalies if total_real_anomalies > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        passed = f1 > 0.3
        score = f1
        
        return BenchmarkResult(
            test_name="Real-World Scenario (工业传感器模拟)",
            passed=passed,
            score=score,
            details={
                'correct_detections': correct_detections,
                'missed_detections': missed_detections,
                'false_alarms': false_alarms,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
            }
        )
    
    def generate_report(self) -> Dict[str, Any]:
        """生成综合报告."""
        passed_count = sum(1 for r in self.results if r.passed)
        total_count = len(self.results)
        avg_score = np.mean([r.score for r in self.results])
        
        report = {
            'summary': {
                'total_tests': total_count,
                'passed_tests': passed_count,
                'pass_rate': passed_count / total_count if total_count > 0 else 0,
                'average_score': avg_score,
            },
            'tests': [asdict(r) for r in self.results],
        }
        
        print("\n" + "="*70)
        print("综合报告")
        print("="*70)
        print(f"通过率: {passed_count}/{total_count} ({100*passed_count/total_count:.1f}%)")
        print(f"平均得分: {avg_score:.2f}")
        
        # H2Q 特性总结
        print("\n" + "-"*70)
        print("H2Q System Control Core 特性验证")
        print("-"*70)
        
        size_result = next((r for r in self.results if 'Model Size' in r.test_name), None)
        if size_result:
            print(f"✓ 极小模型尺寸: {size_result.details.get('size_bytes', 'N/A')} bytes")
        
        det_result = next((r for r in self.results if 'Determinism' in r.test_name), None)
        if det_result:
            print(f"✓ 确定性输出: {'100%' if det_result.passed else 'Failed'}")
        
        lat_result = next((r for r in self.results if 'Latency' in r.test_name), None)
        if lat_result:
            print(f"✓ 实时性能: {lat_result.details.get('mean_latency_us', 'N/A'):.2f} μs")
        
        stab_result = next((r for r in self.results if 'Stability' in r.test_name), None)
        if stab_result:
            print(f"✓ 吞吐量: {stab_result.details.get('throughput_steps_per_sec', 'N/A'):.0f} steps/sec")
        
        return report


def main():
    benchmark = LightweightBenchmark()
    report = benchmark.run_all()
    
    # 保存报告
    with open('lightweight_control_benchmark_results.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n报告已保存到: lightweight_control_benchmark_results.json")
    
    return report


if __name__ == "__main__":
    main()
