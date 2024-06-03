#ifndef TCGMBLAS_GPU_RUNTIME_HPP
#define TCGMBLAS_GPU_RUNTIME_HPP

#pragma warning(disable:2282)
#pragma warning(disable:815 858) 
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#define THREADS_PER_BLOCK 512

#define get_cuda_error(arg) get_cuda_ERROR(arg, __FILE__, __LINE__);
#define get_cublas_error(arg) get_cublas_ERROR(arg, __FILE__, __LINE__);

namespace tcgmtensor{
    class CudaRuntime {
    protected:
        int cudaDevice = -1;
        cudaStream_t stream_ = 0;
        bool async_;
        int version = 0;
        unsigned int streamFlag_ = cudaStreamDefault;
        void createHandle();
        void createStream();
        int blockSize_ = THREADS_PER_BLOCK;
    public:
        
        CudaRuntime();
        CudaRuntime(size_t max_dim, size_t n_mat);
        CudaRuntime(size_t requestedMem);
        ~CudaRuntime();

        bool asyncCopy() {return async_;};
        cublasHandle_t handle = nullptr;
        void print_cuda_version();
        void enableAsyncCopy();
        inline size_t device_id() {return cudaDevice;};
        const inline size_t device_id() const {return cudaDevice;};
        inline void setblockSize(int blockSize) {blockSize_ = blockSize;}
        inline int blockSize() const {return blockSize_;};
        inline int gridSize(size_t base, size_t exp) const {return (int)ceil((float)std::pow(base, exp)/blockSize_);};
        size_t get_GPU_wmaxMem();
        cudaStream_t getStream() {return stream;}
        void synchronize() {get_cuda_error(cudaDeviceSynchronize();)}
    };

    void get_cuda_ERROR(cudaError_t stat, const char * file, int line);
    void get_cublas_ERROR(cublasStatus_t stat, const char* file, int line);
}
#endif