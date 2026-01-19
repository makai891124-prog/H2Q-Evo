#include <metal_stdlib>
using namespace metal;

/**
 * H2Q Lie Algebra Exponential Map Kernel
 * 
 * ARCHITECTURE: SU(2) Quaternionic Manifold (S^3)
 * OPERATION: exp(Ω) : so(3) -> SU(2)
 * HARDWARE: Optimized for M4 GPU/AMX via 16x16 Tiling
 * 
 * [STABLE CODE]
 */

kernel void lie_exponential_map_tiled(
    device const float3* omega [[buffer(0)]],      // Input: Infinitesimal rotation vectors (3D)
    device float4* out_q [[buffer(1)]],            // Output: Unit Quaternions (4D)
    constant uint2& grid_size [[buffer(2)]],       // Grid dimensions for bounds checking
    uint2 gid [[thread_position_in_grid]]          // 2D Grid ID for 16x16 tiling
) {
    // 1. Boundary Guard
    if (gid.x >= grid_size.x || gid.y >= grid_size.y) return;
    
    // 2. Linear Index Calculation
    uint idx = gid.y * grid_size.x + gid.x;
    
    // 3. Atom: Vector Extraction
    float3 w = omega[idx];
    float theta_sq = dot(w, w);
    float theta = sqrt(theta_sq);
    
    // 4. Atom: Sinc/Cos Calculation (Numerical Stability)
    // Using Taylor expansion for small theta to avoid division by zero (infinitesimal limit)
    float s;
    float c;
    
    if (theta < 1e-5f) {
        // Taylor: sin(theta)/theta ≈ 1 - theta^2 / 6
        s = 1.0f - theta_sq / 6.0f;
        // Taylor: cos(theta) ≈ 1 - theta^2 / 2
        c = 1.0f - theta_sq / 2.0f;
    } else {
        s = sin(theta) / theta;
        c = cos(theta);
    }
    
    // 5. Symmetry: Map to S^3 Manifold
    // Resulting Quaternion q = [cos(theta), (sin(theta)/theta) * w]
    // We store as float4(w, x, y, z) where w is the scalar part
    out_q[idx] = float4(c, w.x * s, w.y * s, w.z * s);
}
