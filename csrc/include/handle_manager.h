#pragma once

#include <custatevec.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <memory>
#include <mutex>

namespace torchqml {

class HandleManager {
public:
    static HandleManager& instance();
    
    custatevecHandle_t get_custatevec_handle();
    cublasHandle_t get_cublas_handle();
    
    // Prevent copy/move
    HandleManager(const HandleManager&) = delete;
    HandleManager& operator=(const HandleManager&) = delete;

private:
    HandleManager();
    ~HandleManager();
    
    custatevecHandle_t custatevec_handle_;
    cublasHandle_t cublas_handle_;
    
    // Ensure thread safety for initialization (though static local is thread-safe in C++11)
    // We might need thread-local handles if we want multi-threaded CPU processing driving GPU?
    // For now, assume single stream / standard usage or let CUDA handle serialization.
    // Actually, custatevec handle is not thread-safe if used concurrently.
    // But basic Python execution is GIL locked usually, or single stream.
    // To be safe and simple: Global handle.
};

} // namespace torchqml
