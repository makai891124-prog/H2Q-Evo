#include <metal_stdlib>
using namespace metal;

// H2Q Framework: Hamilton Product AMX-Tiled Kernel
// Optimized for M4 (MPS/16GB) - 1M+ Context Streaming
// Isomorphism: SU(2) <-> S3. Learning as su(2) rotations.

struct Quaternion {
    half4 data; // [r, i, j, k]
};

// Rigid Construction: Hamilton Product Atom
inline half4 hamilton_multiply(half4 q1, half4 q2) {
    return half4(
        q1.x*q2.x - q1.y*q2.y - q1.z*q2.z - q1.w*q2.w, // Real
        q1.x*q2.y + q1.y*q2.x + q1.z*q2.w - q1.w*q2.z, // i
        q1.x*q2.z - q1.y*q2.w + q1.z*q2.x + q1.w*q2.y, // j
        q1.x*q2.w + q1.y*q2.z - q1.z*q2.y + q1.w*q2.x  // k
    );
}

kernel void hamilton_product_tiled(
    device const half4 *A [[buffer(0)]], 
    device const half4 *B [[buffer(1)]],
    device half4 *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    threadgroup half4 *tileA [[threadgroup(0)]],
    threadgroup half4 *tileB [[threadgroup(1)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tpg [[threads_per_threadgroup]],
    uint2 group_id [[threadgroup_position_in_grid]]
) {
    // Elastic Extension: Tiling for 1M+ Context
    // M4 AMX units prefer 32x32 or 64x64 blocks. 
    // We use 16x16 tiles to stay within 16GB L1/L2 cache pressure constraints.
    
    const uint TILE_SIZE = 16;
    half4 accumulation = half4(0.0h);

    for (uint k_tile = 0; k_tile < (K + TILE_SIZE - 1) / TILE_SIZE; k_tile++) {
        // Load tiles into Threadgroup Memory (SRAM)
        uint a_idx = gid.y * K + (k_tile * TILE_SIZE + tid.x);
        uint b_idx = (k_tile * TILE_SIZE + tid.y) * N + gid.x;

        tileA[tid.y * TILE_SIZE + tid.x] = (gid.y < M && (k_tile * TILE_SIZE + tid.x) < K) ? A[a_idx] : half4(0.0h);
        tileB[tid.y * TILE_SIZE + tid.x] = (gid.x < N && (k_tile * TILE_SIZE + tid.y) < K) ? B[b_idx] : half4(0.0h);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute Hamilton Product over the tile
        for (uint k = 0; k < TILE_SIZE; k++) {
            half4 qA = tileA[tid.y * TILE_SIZE + k];
            half4 qB = tileB[k * TILE_SIZE + tid.x];
            accumulation = accumulation + hamilton_multiply(qA, qB);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (gid.y < M && gid.x < N) {
        C[gid.y * N + gid.x] = accumulation;
    }
}

// Spectral Shift Tracker (η) Integration Atom
// η = (1/π) arg{det(S)}
kernel void calculate_spectral_shift(
    device const half4 *S [[buffer(0)]],
    device float *eta [[buffer(1)]],
    constant uint &size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    // Simplified phase deflection calculation for su(2) rotation
    half4 q = S[gid];
    float phase = atan2((float)length(q.yzw), (float)q.x);
    eta[gid] = phase / 3.1415926535f;
}