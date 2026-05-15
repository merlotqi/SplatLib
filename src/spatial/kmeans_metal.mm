/**
 * Metal Backend Implementation for GPU Compute
 */

#ifdef __APPLE__

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <splat/gpu/gpu_compute.h>
#include <iostream>
#include <memory>

namespace splat {
namespace gpu {

class MetalComputeContext {
public:
    static MetalComputeContext& getInstance() {
        static MetalComputeContext instance;
        return instance;
    }
    
    bool isAvailable() const { return device != nullptr; }
    
    id<MTLDevice> getDevice() { return device; }
    id<MTLCommandQueue> getCommandQueue() { return commandQueue; }
    id<MTLLibrary> getLibrary() { return library; }
    
private:
    MetalComputeContext() {
        device = MTLCreateSystemDefaultDevice();
        if (device) {
            commandQueue = [device newCommandQueue];
            
            // Load the Metal library from the application bundle
            NSError *error = nil;
            NSBundle *bundle = [NSBundle bundleForClass:[NSObject class]];
            
            // Try to load the compiled Metal library
            NSURL *libraryURL = [bundle URLForResource:@"kmeans" withExtension:@"metallib"];
            if (libraryURL) {
                library = [device newLibraryWithURL:libraryURL error:&error];
            } else {
                // Fallback: compile from source if available
                // This is typically used in development
                std::cerr << "Warning: Could not find kmeans.metallib bundle resource\n";
            }
        }
    }
    
    ~MetalComputeContext() {
        if (commandQueue) [commandQueue release];
        if (library) [library release];
        if (device) [device release];
    }
    
    id<MTLDevice> device = nullptr;
    id<MTLCommandQueue> commandQueue = nullptr;
    id<MTLLibrary> library = nullptr;
};

bool computeCentroidNorms(const float* centroids, uint32_t K, uint32_t D, float* norms) {
    auto& context = MetalComputeContext::getInstance();
    if (!context.isAvailable()) {
        std::cerr << "Metal device not available\n";
        return false;
    }
    
    id<MTLDevice> device = context.getDevice();
    id<MTLCommandQueue> commandQueue = context.getCommandQueue();
    id<MTLLibrary> library = context.getLibrary();
    
    if (!library) {
        std::cerr << "Metal library not loaded\n";
        return false;
    }
    
    // Create compute pipeline
    id<MTLFunction> kernelFunction = [library newFunctionWithName:@"computeCentroidNorms"];
    if (!kernelFunction) {
        std::cerr << "Could not find computeCentroidNorms kernel\n";
        return false;
    }
    
    NSError *error = nil;
    id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:kernelFunction error:&error];
    [kernelFunction release];
    
    if (!pipelineState) {
        std::cerr << "Failed to create compute pipeline: " << [[error description] UTF8String] << "\n";
        return false;
    }
    
    // Create buffers
    id<MTLBuffer> centroidsBuffer = [device newBufferWithBytes:centroids
                                                         length:K * D * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
    id<MTLBuffer> normsBuffer = [device newBufferWithLength:K * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
    
    // Create command buffer and encoder
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:pipelineState];
    [encoder setBuffer:centroidsBuffer offset:0 atIndex:0];
    [encoder setBuffer:normsBuffer offset:0 atIndex:1];
    [encoder setBytes:&K length:sizeof(uint32_t) atIndex:2];
    [encoder setBytes:&D length:sizeof(uint32_t) atIndex:3];
    
    // Dispatch threads
    MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
    MTLSize gridSize = MTLSizeMake((K + 255) / 256, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Copy results
    memcpy(norms, [normsBuffer contents], K * sizeof(float));
    
    // Cleanup
    [normsBuffer release];
    [centroidsBuffer release];
    [pipelineState release];
    
    return true;
}

bool assignPointsToCentroids(const float* points, const float* centroids, 
                            const float* centroid_norms,
                            uint32_t N, uint32_t K, uint32_t D,
                            uint32_t* results)
{
    auto& context = MetalComputeContext::getInstance();
    if (!context.isAvailable()) {
        std::cerr << "Metal device not available\n";
        return false;
    }
    
    id<MTLDevice> device = context.getDevice();
    id<MTLCommandQueue> commandQueue = context.getCommandQueue();
    id<MTLLibrary> library = context.getLibrary();
    
    if (!library) {
        std::cerr << "Metal library not loaded\n";
        return false;
    }
    
    // Create compute pipeline
    id<MTLFunction> kernelFunction = [library newFunctionWithName:@"assignPointsToCentroids"];
    if (!kernelFunction) {
        std::cerr << "Could not find assignPointsToCentroids kernel\n";
        return false;
    }
    
    NSError *error = nil;
    id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:kernelFunction error:&error];
    [kernelFunction release];
    
    if (!pipelineState) {
        std::cerr << "Failed to create compute pipeline: " << [[error description] UTF8String] << "\n";
        return false;
    }
    
    // Create buffers
    id<MTLBuffer> pointsBuffer = [device newBufferWithBytes:points
                                                     length:N * D * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> centroidsBuffer = [device newBufferWithBytes:centroids
                                                        length:K * D * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
    id<MTLBuffer> normsBuffer = [device newBufferWithBytes:(void*)centroid_norms
                                                    length:K * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> resultsBuffer = [device newBufferWithLength:N * sizeof(uint32_t)
                                                      options:MTLResourceStorageModeShared];
    
    // Create command buffer and encoder
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:pipelineState];
    [encoder setBuffer:pointsBuffer offset:0 atIndex:0];
    [encoder setBuffer:centroidsBuffer offset:0 atIndex:1];
    [encoder setBuffer:normsBuffer offset:0 atIndex:2];
    [encoder setBuffer:resultsBuffer offset:0 atIndex:3];
    [encoder setBytes:&N length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&K length:sizeof(uint32_t) atIndex:5];
    [encoder setBytes:&D length:sizeof(uint32_t) atIndex:6];
    
    // Dispatch threads
    MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
    MTLSize gridSize = MTLSizeMake((N + 255) / 256, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Copy results
    memcpy(results, [resultsBuffer contents], N * sizeof(uint32_t));
    
    // Cleanup
    [resultsBuffer release];
    [normsBuffer release];
    [centroidsBuffer release];
    [pointsBuffer release];
    [pipelineState release];
    
    return true;
}

const char* getBackendName() {
    return "Metal";
}

}  // namespace gpu
}  // namespace splat

#endif  // __APPLE__
