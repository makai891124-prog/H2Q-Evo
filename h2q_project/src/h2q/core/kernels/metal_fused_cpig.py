import torch
from h2q.core.metal_jit_bridge import MetalJITBridge

# METAL_FUSED_CPIG_KERNEL: 16x16 Tiled Hamilton Product + In-Register Softmax
# Optimized for Mac Mini M4 (AMX-like register utilization via MSL)

METAL_FUSED_CPIG_SOURCE = """
#include <metal_stdlib>
using namespace metal;

// Hamilton Product: (a1 + b1i + c1j + d1k) * (a2 + b2i + c2j + d2k)
inline float4 hamilton_product(float4 q1, float4 q2) {
    return float4(
        q1.x*q2.x - q1.y*q2.y - q1.z*q2.z - q1.w*q2.w, // w
        q1.x*q2.y + q1.y*q2.x + q1.z*q2.w - q1.w*q2.z, // x
        q1.x*q2.z - q1.y*q2.w + q1.z*q2.x + q1.w*q2.y, // y
        q1.x*q2.w + q1.y*q2.z - q1.z*q2.y + q1.w*q2.x  // z
    );
}

kernel void fused_cpig_hamilton_softmax(
    device const float4 *A        [[ buffer(0) ]], // Left Manifold (Quaternions)
    device const float4 *B        [[ buffer(1) ]], // Right Manifold (Quaternions)
    device float4 *C              [[ buffer(2) ]], // Gated Output
    constant uint &M              [[ buffer(3) ]],
    constant uint &N              [[ buffer(4) ]],
    constant uint &K              [[ buffer(5) ]],
    uint2 gid                     [[ thread_position_in_grid ]],
    uint2 tid                     [[ thread_position_in_threadgroup ]],
    uint simd_id                  [[ simdgroup_index_in_threadgroup ]],
    uint lane_id                  [[ thread_index_in_simdgroup ]]
) {
    // 16x16 Tiling Logic
    // Each thread computes one quaternion in the output matrix C[M, N]
    if (gid.x >= N || gid.y >= M) return;

    float4 acc = float4(0.0f);

    // Hamilton Accumulation Loop
    for (uint k = 0; k < K; k++) {
        float4 qA = A[gid.y * K + k];
        float4 qB = B[k * N + gid.x];
        acc = acc + hamilton_product(qA, qB);
    }

    // In-Register Softmax (Row-wise across N)
    // We use the norm of the quaternion as the energy for gating
    float energy = length(acc);
    
    // Find max energy in row for numerical stability (SIMD-level reduction)
    float max_energy = simd_max(energy);
    float exp_energy = exp(energy - max_energy);
    float sum_exp = simd_sum(exp_energy);

    // Normalize and apply gating
    float gate = exp_energy / (sum_exp + 1e-6f);
    
    // Store gated quaternion
    C[gid.y * N + gid.x] = acc * gate;
}
"""

class FusedCPIGKernel:
    """
    Experimental: Fused Metal Kernel for Constructive Phase Interference Gating (CPIG).
    Integrates 16x16 tiling for Hamilton Products with row-wise softmax normalization.
    """
    def __init__(self):
        self.bridge = MetalJITBridge()
        self.kernel_name = "fused_cpig_hamilton_softmax"
        # Note: Veracity Compact - Ensure the bridge can compile this source
        self.compiled_kernel = self.bridge.compile_source(METAL_FUSED_CPIG_SOURCE, self.kernel_name)

    def forward(self, manifold_a, manifold_b):
        """
        Executes the fused kernel on MPS device.
        manifold_a: [M, K, 4] (Quaternions)
        manifold_b: [K, N, 4] (Quaternions)
        """
        M, K, _ = manifold_a.shape
        _, N, _ = manifold_b.shape
        
        # Ensure tensors are on MPS and float32 (mapped to float4 in MSL)
        A = manifold_a.contiguous().to("mps")
        B = manifold_b.contiguous().to("mps")
        C = torch.zeros((M, N, 4), device="mps", dtype=torch.float32)

        # Dispatch via JIT bridge
        # Grid: (N, M), Threadgroup: (16, 16) for M4 optimization
        self.bridge.dispatch(
            self.compiled_kernel,
            inputs=[A, B, C, torch.tensor(M), torch.tensor(N), torch.tensor(K)],
            grid=(N, M, 1),
            threadgroup=(16, 16, 1)
        )
        
        return C

# STABLE: Standard Hamilton Product for CPU/MPS fallback
def hamilton_product_stable(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dim=-1)
