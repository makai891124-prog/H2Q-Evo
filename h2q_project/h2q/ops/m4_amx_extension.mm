#include <torch/extension.h>
#include <Metal/Metal.h>
#include <iostream>

// --- METAL SHADER SOURCE (Embedded for JIT Compilation) ---
static const char* HAMILTON_KERNEL_SRC = R"(
#include <metal_stdlib>
using namespace metal;

struct Quaternion {
    float4 val; // [w, i, j, k]
};

// Fused Hamilton Product: q1 * q2
inline float4 hamilton_mul(float4 q1, float4 q2) {
    return float4(
        q1.x*q2.x - q1.y*q2.y - q1.z*q2.z - q1.w*q2.w,
        q1.x*q2.y + q1.y*q2.x + q1.z*q2.w - q1.w*q2.z,
        q1.x*q2.z - q1.y*q2.w + q1.z*q2.x + q1.w*q2.y,
        q1.x*q2.w + q1.y*q2.z - q1.z*q2.y + q1.w*q2.x
    );
}

kernel void fused_hamilton_mm(
    device const float4* A [[buffer(0)]],
    device const float4* B [[buffer(1)]],
    device float4* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) 
{
    if (gid.x >= N || gid.y >= M) return;

    float4 acc = float4(0.0f);
    for (uint k = 0; k < K; k++) {
        float4 a = A[gid.y * K + k];
        float4 b = B[k * N + gid.x];
        acc = acc + hamilton_mul(a, b); // Linear combination of rotations
    }
    C[gid.y * N + gid.x] = acc;
}
)";

// --- C++ INTERFACE ---

void launch_hamilton_kernel(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    auto device = at::msc::getCurrentMetalsDevice();
    auto queue = at::msc::getCurrentMetalsCommandQueue();
    
    NSError* error = nil;
    id<MTLDevice> mtlDevice = (id<MTLDevice>)device;
    id<MTLLibrary> library = [mtlDevice newLibraryWithSource:[NSString stringWithUTF8String:HAMILTON_KERNEL_SRC] 
                                                     options:nil 
                                                       error:&error];
    if (!library) {
        std::cerr << "Failed to compile Hamilton Kernel: " << [[error localizedDescription] UTF8String] << std::endl;
        return;
    }

    id<MTLFunction> function = [library newFunctionWithName:@"fused_hamilton_mm"];
    id<MTLComputePipelineState> pipeline = [mtlDevice newComputePipelineStateWithFunction:function error:&error];

    uint32_t M = a.size(0);
    uint32_t K = a.size(1);
    uint32_t N = b.size(1);

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(id<MTLBuffer>)at::msc::getMTLBufferStorage(a) offset:a.storage_offset() * a.element_size() atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)at::msc::getMTLBufferStorage(b) offset:b.storage_offset() * b.element_size() atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)at::msc::getMTLBufferStorage(c) offset:c.storage_offset() * c.element_size() atIndex:2];
    [encoder setBytes:&M length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&N length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&K length:sizeof(uint32_t) atIndex:5];

    MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
    MTLSize threadGroups = MTLSizeMake((N + 15) / 16, (M + 15) / 16, 1);

    [encoder dispatchThreadgroups:threadGroups threadsPerThreadgroup:threadGroupSize];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_hamilton_kernel", &launch_hamilton_kernel, "M4 Fused Hamilton Kernel");
}
