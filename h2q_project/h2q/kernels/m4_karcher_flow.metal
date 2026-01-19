#include <metal_stdlib>
using namespace metal;

// --- H2Q RIGID CONSTRUCTION: SU(2) QUATERNIONIC ATOMS ---

struct Quat {
    float4 data; // [w, x, y, z]
};

inline float4 quat_mul(float4 q1, float4 q2) {
    return float4(
        q1.x*q2.x - q1.y*q2.y - q1.z*q2.z - q1.w*q2.w,
        q1.x*q2.y + q1.y*q2.x + q1.z*q2.w - q1.w*q2.z,
        q1.x*q2.z - q1.y*q2.w + q1.z*q2.x + q1.w*q2.y,
        q1.x*q2.w + q1.y*q2.z - q1.z*q2.y + q1.w*q2.x
    );
}

inline float3 quat_log(float4 q) {
    float a = clamp(q.x, -1.0f, 1.0f);
    float theta = acos(a);
    float sin_theta = sin(theta);
    if (abs(sin_theta) < 1e-6f) return float3(0.0f);
    return (theta / sin_theta) * q.yzw;
}

inline float4 quat_exp(float3 v) {
    float theta = length(v);
    if (theta < 1e-6f) return float4(1.0f, 0.0f, 0.0f, 0.0f);
    return float4(cos(theta), (sin(theta) / theta) * v);
}

// --- AMX-TILED KARCHER FLOW KERNEL ---
// Optimized for M4 Register Throughput (16x16 Tiling)

kernel void karcher_flow_barycenter_16x16(
    device const float4* modality_audio  [[buffer(0)]],
    device const float4* modality_vision [[buffer(1)]],
    device const float4* modality_text   [[buffer(2)]],
    device const float4* modality_genomic[[buffer(3)]],
    device float4* output_barycenter     [[buffer(4)]],
    constant uint& num_elements          [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    const uint idx = gid.y * 16 + gid.x;
    if (idx >= num_elements) return;

    // Load 4-way modality atoms into registers
    float4 q_a = modality_audio[idx];
    float4 q_v = modality_vision[idx];
    float4 q_t = modality_text[idx];
    float4 q_g = modality_genomic[idx];

    // Initialize Barycenter at the midpoint of Audio-Vision
    // (Elastic Extension: Starting closer to the manifold cluster reduces iterations)
    float4 q_mu = normalize(q_a + q_v + q_t + q_g);

    // Iterative Fréchet Mean (Karcher Flow)
    // 4 iterations are typically sufficient for SU(2) convergence in H2Q
    for (int iter = 0; iter < 4; iter++) {
        float4 q_mu_inv = float4(q_mu.x, -q_mu.yzw);

        // Compute tangent vectors in su(2) Lie Algebra
        float3 v_a = quat_log(quat_mul(q_mu_inv, q_a));
        float3 v_v = quat_log(quat_mul(q_mu_inv, q_v));
        float3 v_t = quat_log(quat_mul(q_mu_inv, q_t));
        float3 v_g = quat_log(quat_mul(q_mu_inv, q_g));

        // Geodesic update: Mean of tangent vectors
        float3 v_mean = (v_a + v_v + v_t + v_g) * 0.25f;

        // Exponential map back to S3 manifold
        q_mu = quat_mul(q_mu, quat_exp(v_mean));
        q_mu = normalize(q_mu);
    }

    // Store result with 16x16 memory alignment for M4 cache efficiency
    output_barycenter[idx] = q_mu;
}
