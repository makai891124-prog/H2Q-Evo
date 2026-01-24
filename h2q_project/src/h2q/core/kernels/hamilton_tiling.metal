#include <metal_stdlib>
using namespace metal;

// [STABLE] Hamilton Product Atom
// Implements the non-commutative quaternionic multiplication: q1 * q2
inline float4 hamilton_product(float4 q1, float4 q2) {
    return float4(
        q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
        q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
        q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w,
        q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
    );
}

// [EXPERIMENTAL] Hamilton-AMX Tiling Kernel
// Optimized for M4 Silicon (120GB/s bandwidth target)
// Tiling: 16x16 Quaternionic Grid (64x64 float scalar equivalent)
kernel void hamilton_quaternion_matmul(
    device const float4* matrix_A [[buffer(0)]], 
    device const float4* matrix_B [[buffer(1)]],
    device float4* matrix_C       [[buffer(2)]],
    constant uint& dim_K          [[buffer(3)]],
    uint2 gid                     [[thread_position_in_grid]],
    uint2 tid                     [[thread_position_in_threadgroup]],
    uint2 lid                     [[thread_index_in_threadgroup]])
{
    const uint TILE_SIZE = 16;
    
    // Threadgroup memory for L1 cache optimization
    threadgroup float4 tile_A[16][16];
    threadgroup float4 tile_B[16][16];

    float4 accumulator = float4(0.0f);

    // Loop over the K-dimension in chunks of TILE_SIZE
    for (uint k = 0; k < dim_K; k += TILE_SIZE) {
        
        // Load data into threadgroup memory (Symmetric Loading)
        // matrix_A is [M, K], matrix_B is [K, N]
        tile_A[tid.y][tid.x] = matrix_A[gid.y * dim_K + (k + tid.x)];
        tile_B[tid.y][tid.x] = matrix_B[(k + tid.y) * dim_K + gid.x];

        // Synchronize to ensure tiles are fully loaded
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute Hamilton products for the tile
        #pragma unroll
        for (uint i = 0; i < TILE_SIZE; i++) {
            accumulator = hamilton_product(tile_A[tid.y][i], tile_B[i][tid.x]);
        }

        // Synchronize before next tile load
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result to global memory (L0 Topological Spelling)
    // Mapping the 256-dim manifold back to global coordinates
    matrix_C[gid.y * dim_K + gid.x] = accumulator;
}
