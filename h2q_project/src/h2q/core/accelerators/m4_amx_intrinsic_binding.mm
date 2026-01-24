#include <torch/extension.h>
#include <ATen/mps/MPSDevice.h>
#include <Metal/Metal.h>
#include <iostream>

// --- M4 AMX HAMILTON KERNEL (MSL) ---
static const char* HAMILTON_AMX_SRC = R"(
#include <metal_stdlib>
using namespace metal;

// 16x16 Tiled Hamilton Product using simdgroup_matrix intrinsics
// Layout: [N, M, 4] where 4 represents (1, i, j, k)
kernel void hamilton_gemm_amx_16x16(
    device const float* matA [[buffer(0)]],
    device const float* matB [[buffer(1)]],
    device float* matOut      [[buffer(2)]],
    constant uint& M          [[buffer(3)]],
    constant uint& N          [[buffer(4)]],
    constant uint& K          [[buffer(5)]],
    uint2 gid [[threadblock_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    // Each simdgroup handles a 16x16 tile
    // Quaternionic GEMM requires 16 real-valued matrix multiplications
    // We load 4 components (a, b, c, d) for A and (x, y, z, w) for B
    
    simdgroup_matrix<float, 16, 16> out_r(0), out_i(0), out_j(0), out_k(0);
    
    for (uint k = 0; k < K; k += 16) {
        // Load tiles for components
        simdgroup_matrix<float, 16, 16> a_r, a_i, a_j, a_k;
        simdgroup_matrix<float, 16, 16> b_r, b_i, b_j, b_k;
        
        // Strided loads for quaternionic components
        simdgroup_load(a_r, matA + (gid.y * 16 * K + k) * 4 + 0, K * 4, uint2(0));
        simdgroup_load(a_i, matA + (gid.y * 16 * K + k) * 4 + 1, K * 4, uint2(0));
        simdgroup_load(a_j, matA + (gid.y * 16 * K + k) * 4 + 2, K * 4, uint2(0));
        simdgroup_load(a_k, matA + (gid.y * 16 * K + k) * 4 + 3, K * 4, uint2(0));
        
        simdgroup_load(b_r, matB + (k * N + gid.x * 16) * 4 + 0, N * 4, uint2(0));
        simdgroup_load(b_i, matB + (k * N + gid.x * 16) * 4 + 1, N * 4, uint2(0));
        simdgroup_load(b_j, matB + (k * N + gid.x * 16) * 4 + 2, N * 4, uint2(0));
        simdgroup_load(b_k, matB + (k * N + gid.x * 16) * 4 + 3, N * 4, uint2(0));

        // Hamilton Product Logic: 
        // Real: ar*br - ai*bi - aj*bj - ak*bk
        out_r = simdgroup_multiply_accumulate(a_r, b_r, out_r);
        out_r = simdgroup_multiply_accumulate(-a_i, b_i, out_r);
        out_r = simdgroup_multiply_accumulate(-a_j, b_j, out_r);
        out_r = simdgroup_multiply_accumulate(-a_k, b_k, out_r);

        // Imag (i): ar*bi + ai*br + aj*bk - ak*bj
        out_i = simdgroup_multiply_accumulate(a_r, b_i, out_i);
        out_i = simdgroup_multiply_accumulate(a_i, b_r, out_i);
        out_i = simdgroup_multiply_accumulate(a_j, b_k, out_i);
        out_i = simdgroup_multiply_accumulate(-a_k, b_j, out_i);

        // Imag (j): ar*bj - ai*bk + aj*br + ak*bi
        out_j = simdgroup_multiply_accumulate(a_r, b_j, out_j);
        out_j = simdgroup_multiply_accumulate(-a_i, b_k, out_j);
        out_j = simdgroup_multiply_accumulate(a_j, b_r, out_j);
        out_j = simdgroup_multiply_accumulate(a_k, b_i, out_j);

        // Imag (k): ar*bk + ai*bj - aj*bi + ak*br
        out_k = simdgroup_multiply_accumulate(a_r, b_k, out_k);
        out_k = simdgroup_multiply_accumulate(a_i, b_j, out_k);
        out_k = simdgroup_multiply_accumulate(-a_j, b_i, out_k);
        out_k = simdgroup_multiply_accumulate(a_k, b_r, out_k);
    }

    // Store results back to quaternionic layout
    simdgroup_store(out_r, matOut + (gid.y * 16 * N + gid.x * 16) * 4 + 0, N * 4, uint2(0));
    simdgroup_store(out_i, matOut + (gid.y * 16 * N + gid.x * 16) * 4 + 1, N * 4, uint2(0));
    simdgroup_store(out_j, matOut + (gid.y * 16 * N + gid.x * 16) * 4 + 2, N * 4, uint2(0));
    simdgroup_store(out_k, matOut + (gid.y * 16 * N + gid.x * 16) * 4 + 3, N * 4, uint2(0));
}
)";

// --- PYTORCH BINDING ---

at::Tensor hamilton_gemm_amx(at::Tensor matA, at::Tensor matB) {
    TORCH_CHECK(matA.is_mps(), "matA must be an MPS tensor");
    TORCH_CHECK(matB.is_mps(), "matB must be an MPS tensor");
    TORCH_CHECK(matA.size(2) == 4 && matB.size(2) == 4, "Tensors must be quaternionic [N, M, 4]");

    int M_val = matA.size(0);
    int K_val = matA.size(1);
    int N_val = matB.size(1);

    auto options = at::TensorOptions().device(at::kMPS).dtype(at::kFloat);
    at::Tensor matOut = at::empty({M_val, N_val, 4}, options);

    id<MTLDevice> device = at::mps::get_mtl_device();
    id<MTLCommandQueue> queue = at::mps::get_command_queue();

    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:HAMILTON_AMX_SRC] 
                                                   options:nil 
                                                     error:&error];
    if (!library) {
        std::cerr << "Failed to compile Hamilton AMX Kernel: " << [[error localizedDescription] UTF8String] << std::endl;
        return matOut;
    }

    id<MTLFunction> function = [library newFunctionWithName:@"hamilton_gemm_amx_16x16"];
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:at::mps::get_mtl_buffer_from_tensor(matA) offset:matA.storage_offset() * sizeof(float) atIndex:0];
    [encoder setBuffer:at::mps::get_mtl_buffer_from_tensor(matB) offset:matB.storage_offset() * sizeof(float) atIndex:1];
    [encoder setBuffer:at::mps::get_mtl_buffer_from_tensor(matOut) offset:matOut.storage_offset() * sizeof(float) atIndex:2];
    [encoder setBytes:&N_val length:sizeof(uint) atIndex:3];
    [encoder setBytes:&M_val length:sizeof(uint) atIndex:4];
    [encoder setBytes:&K_val length:sizeof(uint) atIndex:5];

    // Grid size: 16x16 tiles
    MTLSize threadgroupSize = MTLSizeMake(32, 1, 1); // 1 SIMD group per threadgroup
    MTLSize gridSize = MTLSizeMake(N_val / 16, M_val / 16, 1);

    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    return matOut;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hamilton_gemm_amx", &hamilton_gemm_amx, "M4 AMX Accelerated Hamilton GEMM");
}