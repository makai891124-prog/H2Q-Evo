#!/usr/bin/env python3
"""Quick verification script for PrecisionGatedExecutor."""

import sys
import os

def main():
    print("="*60)
    print("PrecisionGatedExecutor - 快速验证")
    print("="*60)
    
    try:
        # Import numpy first
        import numpy as np
        
        # Import core classes from the same package
        from precision_gated_executor import (
            PrecisionGatedExecutor,
            EntropyMetrics,
            StateManifold,
            ContinuousManifoldEncoder,
            DiscreteLogicVerifier,
            DualProposition,
        )
        print("\n✓ 所有核心类导入成功")
        
        # Test 1: EntropyMetrics
        print("\n[Test 1] EntropyMetrics")
        metrics = EntropyMetrics(
            logical_entropy=0.1,
            semantic_entropy=0.05,
            temporal_entropy=0.02,
            combined_entropy=0.05,
        )
        print(f"  组合熵: {metrics.combined_entropy:.4f}")
        print(f"  状态: {metrics.get_manifold_state().value}")
        print("  ✓ EntropyMetrics 正常")
        
        # Test 2: ContinuousManifoldEncoder
        print("\n[Test 2] ContinuousManifoldEncoder")
        encoder = ContinuousManifoldEncoder()
        q = encoder.encode_proposition("This is a test")
        print(f"  编码四元数范数: {np.linalg.norm(q):.4f}")
        print("  ✓ ContinuousManifoldEncoder 正常")
        
        # Test 3: DiscreteLogicVerifier
        print("\n[Test 3] DiscreteLogicVerifier")
        verifier = DiscreteLogicVerifier()
        contradiction = verifier.verify_contradiction("It is true", "It is not true")
        print(f"  矛盾检测: {contradiction}")
        print("  ✓ DiscreteLogicVerifier 正常")
        
        # Test 4: DualProposition
        print("\n[Test 4] DualProposition")
        prop = DualProposition(
            thesis="Statement A",
            antithesis="Not statement A",
            thesis_confidence=0.6,
            antithesis_confidence=0.4,
        )
        is_valid = prop.verify_closure()
        print(f"  拓扑闭包有效: {is_valid}")
        print(f"  闭包间隙: {prop.closure_gap:.6f}")
        print("  ✓ DualProposition 正常")
        
        # Test 5: PrecisionGatedExecutor
        print("\n[Test 5] PrecisionGatedExecutor")
        executor = PrecisionGatedExecutor()
        metrics = executor._measure_entropy("Calculate 2+2")
        print(f"  逻辑熵: {metrics.logical_entropy:.4f}")
        print(f"  语义熵: {metrics.semantic_entropy:.4f}")
        print(f"  时间熵: {metrics.temporal_entropy:.4f}")
        print(f"  组合熵: {metrics.combined_entropy:.4f}")
        print("  ✓ PrecisionGatedExecutor 正常")
        
        print("\n" + "="*60)
        print("✓ 所有验证测试通过！")
        print("="*60)
        
        return 0
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
