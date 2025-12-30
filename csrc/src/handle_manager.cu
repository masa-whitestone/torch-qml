#include "handle_manager.h"
#include <stdexcept>
#include <string>

namespace torchqml {

HandleManager& HandleManager::instance() {
    static HandleManager instance;
    return instance;
}

HandleManager::HandleManager() {
    if (custatevecCreate(&custatevec_handle_) != CUSTATEVEC_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to create cuStateVec handle");
    }
    if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
        custatevecDestroy(custatevec_handle_);
        throw std::runtime_error("Failed to create cuBLAS handle");
    }
}

HandleManager::~HandleManager() {
    if (custatevec_handle_) custatevecDestroy(custatevec_handle_);
    if (cublas_handle_) cublasDestroy(cublas_handle_);
}

custatevecHandle_t HandleManager::get_custatevec_handle() {
    return custatevec_handle_;
}

cublasHandle_t HandleManager::get_cublas_handle() {
    return cublas_handle_;
}

} // namespace torchqml
