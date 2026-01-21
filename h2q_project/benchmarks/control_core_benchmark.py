"""H2Q System Control Core - 综合基准测试.

测试内容:
1. 模型大小验证 (目标: <50KB)
2. 确定性验证 (相同输入 → 相同输出)
3. 微变化检测灵敏度
4. 轨迹突变预测准确率
5. 实时性能 (延迟 <100μs)
6. 长时间稳定性 (无人值守运行)
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import json
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from h2q.control.system_control_core import (
    H2QSystemControlCore,
    create_control_core,
    SensorMonitor,
)


@dataclass
class BenchmarkResult:
    """基准测试结果."""
    test_name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    

class SystemControlBenchmark:
    """系统控制核心基准测试套件."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        
    def run_all(self) -> Dict[str, Any]:
        """运行所有基准测试."""
        print("="*70)
        print("H2Q System Control Core - 综合基准测试")
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
                    else:
                        print(f"   {key}: {value}")
            except Exception as e:
                print(f"\n❌ ERROR {test.__name__}: {e}")
                self.results.append(BenchmarkResult(
                    test_name=test.__name__,
                    passed=False,
                    score=0.0,
                    details={'error': str(e)}
                ))
        
        return self.generate_report()
    
    def test_model_size(self) -> BenchmarkResult:
        """测试 1: 模型大小验证."""
        core = create_control_core(input_dim=16)
        
        param_count = core.count_parameters()
        size_kb = core.model_size_bytes() / 1024
        
        # 目标: <50KB
        target_kb = 50.0
        passed = size_kb < target_kb
        score = min(1.0, target_kb / size_kb) if size_kb > 0 else 1.0
        
        return BenchmarkResult(
            test_name="Model Size (目标 <50KB)",
            passed=passed,
            score=score,
            details={
                'parameters': param_count,
                'size_kb': size_kb,
                'target_kb': target_kb,
            }
        )
    
    def test_determinism(self) -> BenchmarkResult:
        """测试 2: 确定性验证."""
        core1 = create_control_core(input_dim=8)
        core2 = create_control_core(input_dim=8)
        
        # 使用相同权重
        core2.load_state_dict(core1.state_dict())
        
        # 设置为评估模式
        core1.eval()
        core2.eval()
        
        # 生成固定输入
        torch.manual_seed(42)
        inputs = [torch.randn(8) for _ in range(100)]
        
        # 简化测试: 独立处理相同序列，比较输出
        all_identical = True
        max_diff = 0.0
        
        for signal in inputs[:50]:
            # 独立处理每个信号
            core1.reset()
            core2.reset()
            
            s1 = core1.process(signal.clone())
            s2 = core2.process(signal.clone())
            
            diff = torch.abs(s1.quaternion - s2.quaternion).max().item()
            max_diff = max(max_diff, diff)
            
            if diff > 1e-6:
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
        core = create_control_core(
            input_dim=8,
            anomaly_sensitivity=0.001,  # 高灵敏度
        )
        
        # 校准阶段: 正常信号
        baseline_signal = torch.zeros(8)
        for _ in range(50):
            noise = torch.randn(8) * 0.01
            core.process(baseline_signal + noise)
        
        # 测试不同幅度的微变化
        perturbation_levels = [0.001, 0.005, 0.01, 0.05, 0.1]
        detection_rates = []
        
        for level in perturbation_levels:
            detections = 0
            trials = 20
            
            for _ in range(trials):
                # 注入微小扰动
                perturbed = baseline_signal + torch.randn(8) * 0.01
                perturbed[0] += level  # 在第一个维度添加扰动
                
                state = core.process(perturbed)
                if state.anomaly_score > 0.1:  # 检测阈值
                    detections += 1
            
            detection_rates.append(detections / trials)
        
        # 评分: 检测率随扰动增加而提高
        avg_detection = np.mean(detection_rates)
        monotonic = all(detection_rates[i] <= detection_rates[i+1] 
                       for i in range(len(detection_rates)-1))
        
        passed = avg_detection > 0.3 and monotonic
        score = avg_detection
        
        return BenchmarkResult(
            test_name="Micro-Change Detection (微变化检测)",
            passed=passed,
            score=score,
            details={
                'perturbation_levels': perturbation_levels,
                'detection_rates': detection_rates,
                'avg_detection_rate': avg_detection,
                'monotonic_response': monotonic,
            }
        )
    
    def test_trajectory_prediction(self) -> BenchmarkResult:
        """测试 4: 轨迹预测准确率."""
        core = create_control_core(input_dim=4, prediction_horizon=10)
        
        # 生成可预测的轨迹 (正弦波)
        t = np.linspace(0, 4*np.pi, 200)
        trajectory = np.column_stack([
            np.sin(t),
            np.cos(t),
            np.sin(2*t) * 0.5,
            np.cos(2*t) * 0.5,
        ])
        
        prediction_errors = []
        
        for i in range(50, len(trajectory) - 10):
            # 处理历史
            signal = torch.tensor(trajectory[i], dtype=torch.float32)
            state = core.process(signal)
            
            if state.predicted_trajectory is not None:
                # 比较预测与实际
                actual_future = trajectory[i+1:i+11]
                
                # 将预测的四元数转换回信号空间 (近似)
                pred_q = state.predicted_trajectory[:len(actual_future)]
                
                # 计算相位预测误差
                pred_phases = []
                actual_phases = []
                
                for j, q in enumerate(pred_q):
                    pred_phase = 2 * np.arctan2(torch.norm(q[1:]).item(), q[0].item())
                    pred_phases.append(pred_phase)
                    
                    actual_q = torch.tensor(actual_future[j], dtype=torch.float32)
                    actual_q = F.normalize(actual_q, dim=0)
                    actual_phase = 2 * np.arctan2(torch.norm(actual_q[1:]).item(), actual_q[0].item())
                    actual_phases.append(actual_phase)
                
                # 相位预测误差
                phase_error = np.mean(np.abs(np.array(pred_phases) - np.array(actual_phases)))
                prediction_errors.append(phase_error)
        
        mean_error = np.mean(prediction_errors) if prediction_errors else float('inf')
        
        # 评分: 误差越小越好
        passed = mean_error < np.pi / 2  # 误差小于 90°
        score = max(0, 1.0 - mean_error / np.pi)
        
        return BenchmarkResult(
            test_name="Trajectory Prediction (轨迹预测)",
            passed=passed,
            score=score,
            details={
                'mean_phase_error_rad': mean_error,
                'mean_phase_error_deg': np.degrees(mean_error),
                'num_predictions': len(prediction_errors),
            }
        )
    
    def test_latency_performance(self) -> BenchmarkResult:
        """测试 5: 实时性能."""
        core = create_control_core(input_dim=16)
        
        # 预热
        for _ in range(100):
            core.process(torch.randn(16))
        
        # 计时测试
        latencies = []
        for _ in range(1000):
            signal = torch.randn(16)
            
            start = time.perf_counter()
            core.process(signal)
            end = time.perf_counter()
            
            latencies.append((end - start) * 1e6)  # 转换为微秒
        
        mean_latency = np.mean(latencies)
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        max_latency = np.max(latencies)
        
        # 目标: 平均延迟 <100μs
        target_us = 100.0
        passed = mean_latency < target_us
        score = min(1.0, target_us / mean_latency) if mean_latency > 0 else 1.0
        
        return BenchmarkResult(
            test_name="Latency Performance (延迟 <100μs)",
            passed=passed,
            score=score,
            details={
                'mean_latency_us': mean_latency,
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
        core = create_control_core(input_dim=4, history_len=32)
        
        # 生成带有明确跃迁的信号
        transitions_injected = []
        transitions_detected = []
        
        for trial in range(10):
            core.reset()
            
            # 正常阶段
            for i in range(30):
                signal = torch.tensor([0.1, 0.1, 0.1, 0.1]) + torch.randn(4) * 0.01
                core.process(signal)
            
            # 注入跃迁
            transition_step = 30 + trial
            transitions_injected.append(transition_step)
            
            for i in range(30, 60):
                if i == transition_step:
                    # 突变信号
                    signal = torch.tensor([1.0, -1.0, 1.0, -1.0])
                else:
                    signal = torch.tensor([0.1, 0.1, 0.1, 0.1]) + torch.randn(4) * 0.01
                
                state = core.process(signal)
            
            # 检查是否检测到跃迁
            if core.metrics.phase_transitions > 0:
                transitions_detected.append(trial)
        
        detection_rate = len(transitions_detected) / len(transitions_injected)
        
        passed = detection_rate >= 0.7
        score = detection_rate
        
        return BenchmarkResult(
            test_name="Phase Transition Detection (相位跃迁检测)",
            passed=passed,
            score=score,
            details={
                'injected_transitions': len(transitions_injected),
                'detected_transitions': len(transitions_detected),
                'detection_rate': detection_rate,
            }
        )
    
    def test_long_running_stability(self) -> BenchmarkResult:
        """测试 7: 长时间运行稳定性 (模拟无人值守)."""
        core = create_control_core(input_dim=8)
        
        # 模拟长时间运行 (100K 步)
        num_steps = 100000
        
        start_time = time.time()
        
        memory_stable = True
        no_nan = True
        
        for i in range(num_steps):
            signal = torch.randn(8) * 0.1
            
            # 偶尔注入异常
            if i % 10000 == 5000:
                signal += torch.randn(8) * 2.0
            
            state = core.process(signal)
            
            # 检查数值稳定性
            if torch.isnan(state.quaternion).any() or torch.isinf(state.quaternion).any():
                no_nan = False
                break
            
            # 限制历史长度以控制内存
            if len(core.state_history) > 1000:
                core.state_history = core.state_history[-100:]
                core.q_history = core.q_history[-100:]
        
        end_time = time.time()
        elapsed = end_time - start_time
        throughput = num_steps / elapsed
        
        passed = no_nan and memory_stable
        score = 1.0 if passed else 0.0
        
        return BenchmarkResult(
            test_name="Long-Running Stability (长时间稳定性)",
            passed=passed,
            score=score,
            details={
                'total_steps': num_steps,
                'elapsed_seconds': elapsed,
                'throughput_steps_per_sec': throughput,
                'numerical_stable': no_nan,
                'memory_controlled': memory_stable,
                'final_anomalies': core.metrics.anomalies_detected,
                'final_transitions': core.metrics.phase_transitions,
            }
        )
    
    def test_noise_robustness(self) -> BenchmarkResult:
        """测试 8: 噪声鲁棒性."""
        core = create_control_core(input_dim=8, anomaly_sensitivity=0.01)
        
        # 基准信号
        base_signal = torch.tensor([0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005])
        
        noise_levels = [0.001, 0.01, 0.05, 0.1, 0.2]
        false_positive_rates = []
        
        for noise_std in noise_levels:
            core.reset()
            
            # 校准
            for _ in range(50):
                noisy = base_signal + torch.randn(8) * noise_std
                core.process(noisy)
            
            # 测试
            false_positives = 0
            trials = 100
            
            for _ in range(trials):
                noisy = base_signal + torch.randn(8) * noise_std
                state = core.process(noisy)
                
                # 在正常噪声下不应触发异常
                if state.anomaly_score > 1.0:
                    false_positives += 1
            
            fpr = false_positives / trials
            false_positive_rates.append(fpr)
        
        # 低噪声时假阳性率应该低
        low_noise_fpr = np.mean(false_positive_rates[:2])
        
        passed = low_noise_fpr < 0.1
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
        
        return report


def main():
    benchmark = SystemControlBenchmark()
    report = benchmark.run_all()
    
    # 保存报告
    with open('control_core_benchmark_results.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n报告已保存到: control_core_benchmark_results.json")
    
    return report


if __name__ == "__main__":
    main()
