#include <torch/extension.h>
#include <ATen/native/mps/OperationUtils.h>
#include <Metal/Metal.h>
#include <iostream>

/* 
 * H2Q M4 Native Bridge: Quaternionic Geodesic Flow Accelerator
 * Substrate: SU(2) Manifold
 * Optimization: 16x16 Tiled MSL Kernel for Quaternionic GEMM
 */

static const char* QUAT_GEMM_KERNEL = R"( 
#include <metal_stdlib>
using namespace metal;

kernel void quaternionic_gemm_16x16(
    device const float4* A [[buffer(0)]], 
    device const float4* B [[buffer(1)]],
    device float4* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    threadgroup float4 tileA[16][16],
    threadgroup float4 tileB[16][16]
) {
    uint row = gid.y;
    uint col = gid.x;
    float4 acc = float4(0.0f);

    for (uint t = 0; t < (K + 15) / 16; t++) {
        // Load tiles into threadgroup memory (Symmetry Enforcement)
        if (row < M && (t * 16 + tid.x) < K) 
            tileA[tid.y][tid.x] = A[row * K + t * 16 + tid.x];
        else 
            tileA[tid.y][tid.x] = float4(0.0f);

        if (col < N && (t * 16 + tid.y) < K)
            tileB[tid.y][tid.x] = B[(t * 16 + tid.y) * N + col];
        else
            tileB[tid.y][tid.x] = float4(0.0f);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < 16; k++) {
            float4 q1 = tileA[tid.y][k];
            float4 q2 = tileB[k][tid.x];
            
            // Hamilton Product: (a1+b1i+c1j+d1k) * (a2+b2i+c2j+d2k)
            // q.w = real, q.xyz = imaginary (i, j, k)
            acc.w += q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z;
            acc.x += q1.w*q2.x + q1.x*q2.w + q1.y*q2.z - q1.z*q2.y;
            acc.y += q1.w*q2.y - q1.x*q2.z + q1.y*q2.w + q1.z*q2.x;
            acc.z += q1.w*q2.z + q1.x*q2.y - q1.y*q2.x + q1.z*q2.w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}
)";

// Global state for Metal Pipeline
static id<MTLComputePipelineState> pipelineState = nil;
static id<MTLDevice> device = nil;

void initialize_mps_bridge() {
    if (pipelineState) return;
    
    @autoreleasepool {
        device = at::native::mps::getMPSDevice();
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:QUAT_GEMM_KERNEL] 
                                                       options:nil 
                                                         error:&error];
        if (!library) {
            std::cerr << "H2Q Bridge Error: Failed to compile MSL kernel: " << [[error localizedDescription] UTF8String] << std::endl;
            return;
        }
        
        id<MTLFunction> function = [library newFunctionWithName:@"quaternionic_gemm_16x16"];
        pipelineState = [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipelineState) {
            std::cerr << "H2Q Bridge Error: Failed to create pipeline state" << std::endl;
        }
    }
}

torch::Tensor m4_quaternionic_gemm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.device().is_mps(), "A must be on MPS");
    TORCH_CHECK(B.device().is_mps(), "B must be on MPS");
    
    initialize_mps_bridge();

    auto M = A.size(0);
    auto K = A.size(1);
    auto N = B.size(1);
    
    auto C = torch::empty({M, N, 4}, A.options());

    @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = at::native::mps::getMPSStream()->commandBuffer();
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:pipelineState];
        
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(A) offset:A.storage_offset() * A.element_size() atIndex:0];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(B) offset:B.storage_offset() * B.element_size() atIndex:1];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(C) offset:C.storage_offset() * C.element_size() atIndex:2];
        
        uint32_t m_val = (uint32_t)M;
        uint32_t n_val = (uint32_t)N;
        uint32_t k_val = (uint32_t)K;
        
        [encoder setBytes:&m_val length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&n_val length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&k_val length:sizeof(uint32_t) atIndex:5];
        
        MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
        MTLSize gridSize = MTLSizeMake((N + 15) / 16 * 16, (M + 15) / 16 * 16, 1);
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        
        at::native::mps::getMPSStream()->synchronize(at::native::mps::SyncType::COMMIT_AND_CONTINUE);
    }
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quaternionic_gemm", &m4_quaternionic_gemm, "M4 Optimized Quaternionic GEMM (MPS)");
}