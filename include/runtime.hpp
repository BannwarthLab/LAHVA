#ifndef TCGMBLAS_GPU_RUNTIME_HPP
#define TCGMBLAS_GPU_RUNTIME_HPP
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#define THREADS_PER_BLOCK 512

namespace tcgmtensor{
    class CudaRuntime {
    protected:
        size_t cudaDevice = 0;
        cudaStream_t stream = nullptr;
        int version = 0;
        void createHandle();
        const int blockSize_ = THREADS_PER_BLOCK;
    public:
        cublasHandle_t handle = nullptr;
        CudaRuntime();
        CudaRuntime(size_t max_dim, size_t);
        ~CudaRuntime();
        void print_cuda_version();
        inline size_t device_id() {return cudaDevice;};
        const inline size_t device_id() const {return cudaDevice;};
        inline int blockSize() const {return blockSize_;};
        inline int gridSize(size_t ndim, size_t dim) const {return (int)ceil((float)std::pow(ndim, dim)/blockSize_);};

    };
    void get_cuda_error(cudaError_t stat);
    void get_cublas_error(cublasStatus_t stat);
}
#endif