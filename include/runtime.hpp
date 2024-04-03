#ifndef TCGMBLAS_GPU_RUNTIME_HPP
#define TCGMBLAS_GPU_RUNTIME_HPP
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace tcgmtensor{
    class CudaRuntime {
    protected:
        size_t cudaDevice = 0;
        cudaStream_t stream = nullptr;
        int version = 0;
        void createHandle();
    public:
        cublasHandle_t handle = nullptr;
        CudaRuntime();
        CudaRuntime(size_t max_dim, size_t);
        ~CudaRuntime();
        void print_cuda_version();
    };
    void get_cuda_error(cudaError_t stat);
    void get_cublas_error(cublasStatus_t stat);
}
#endif