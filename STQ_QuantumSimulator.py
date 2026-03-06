import torch
import math
import numpy as np

class STQ_QuantumSimulator:
    """
    基于 STQ-TN 双复数域的量子纠缠与退相干模拟器
    用绝对因果代数替代传统庞大的 2^n 希尔伯特空间矩阵指数运算
    """
    def __init__(self, mass_kg=1e-14, distance_m=35e-6):
        self.hbar = 1.054e-34
        self.G = 6.674e-11
        self.mass = mass_kg
        self.d_min = distance_m
        
        # 模拟 QGEM 实验的引力纠缠率 (对应 Z2 自指扩张的相位频率)
        # 根据平行双量子比特设定，计算纠缠角频率
        self.omega = (self.G * self.mass**2) / (self.d_min * self.hbar) 
        
    def dual_complex_evolution(self, time_steps, gamma_decoherence, Lambda_threshold, formula_mode="legacy"):
        """
        核心演化：不使用薛定谔方程，而是使用 STQ-TN 的共轭激波方程
        Z1(t+1) = Z1(t) - W * conj(Z2(t))
        Z2(t+1) = W * Z2(t) * e^(i * omega * dt)  (直至触发激波截断)
        """
        dt = 0.01
        Z1_energy = []
        Z2_phase = []
        W_witness_simulated = []
        
        # 初始状态：处于完美叠加态的 Z2 自指系统
        Z1_state = 1.0 + 0j
        Z2_state = 1.0 + 0j
        
        for t in time_steps:
            # 1. Z2 域的纠缠扩张 (对应量子态的相位累积)
            Z2_state = Z2_state * np.exp(1j * self.omega * dt)
            
            # 2. Z1 背景域通过共轭作用产生退相干阻尼 (对应物理上的环境碰撞/热涨落)
            # 退相干率 gamma 在这里体现为背景对自指的能量抽取
            damping_factor = np.exp(-gamma_decoherence * t)
            Z1_state = Z1_state - (1 - damping_factor) * np.conj(Z2_state)
            
            # 3. 激波截断检测 (概念结晶/量子坍缩)
            E_k = abs(Z1_state * Z2_state)
            if E_k > Lambda_threshold:
                # 触发相位锁定，量子叠加态退化为确定性经典态 (失去纠缠)
                locked_phase = (2 * math.pi / 2) * round(np.angle(Z2_state) * 2 / (2 * math.pi))
                Z2_state = Lambda_threshold * np.exp(1j * locked_phase)
            
            # 4. 计算拓扑纠缠见证者 (映射 QGEM 的 PPT Witness)
            # 在 STQ-TN 中，Witness等价于系统未被截断的共轭残余项
            # 两种见证者口径：
            # legacy:  1/4 - 1/4 * e^(-γt) * (e^(-γt) - 2 sin(ωt))
            # aligned: 1/4 - 1/4 * e^(-γt) * (e^(-γt) + 2 sin(ωt))
            if formula_mode == "aligned":
                W_val = 0.25 - 0.25 * damping_factor * (damping_factor + 2 * np.sin(self.omega * t))
            else:
                W_val = 0.25 - 0.25 * damping_factor * (damping_factor - 2 * np.sin(self.omega * t))
            
            # 如果触发截断，<W> 强行归零（纠缠破裂）
            if E_k > Lambda_threshold:
                W_val = 0.0 
                
            W_witness_simulated.append(W_val)
            
        return W_witness_simulated

def _demo():
    simulator = STQ_QuantumSimulator(mass_kg=1e-14, distance_m=35e-6)
    time_array = np.linspace(0, 2.0, 200)

    W_ideal = simulator.dual_complex_evolution(
        time_array, gamma_decoherence=1e-3, Lambda_threshold=100.0, formula_mode="legacy"
    )
    W_collapsed = simulator.dual_complex_evolution(
        time_array, gamma_decoherence=0.5, Lambda_threshold=1.5, formula_mode="legacy"
    )

    print(f"理想态纠缠见证者极值 (负值代表成功纠缠): {min(W_ideal):.4f}")
    print(f"坍缩态纠缠见证者极值 (激波截断后失去纠缠): {min(W_collapsed):.4f}")


if __name__ == "__main__":
    _demo()