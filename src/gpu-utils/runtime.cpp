#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "runtime.hpp"
#include <iostream>
#include <string>

namespace tcgmtensor
{
    CudaRuntime::CudaRuntime() 
    {  

        createHandle();
    };

    CudaRuntime::~CudaRuntime() {
        if (stream != nullptr) cudaStreamDestroy(stream);
        if (handle != nullptr) cublasDestroy(handle);
        cudaDeviceReset();
    };

    CudaRuntime::CudaRuntime(size_t max_dim, size_t n_mat)
    {
        size_t freemem, totmem;
        int ndev;
        size_t request;
        size_t availmem = 0;

        freemem = 0;
        totmem = 0;
        cudaError_t stat = cudaGetDeviceCount(&ndev);
        get_cuda_error(stat);
        for (int i = 0; i < ndev; ++i) {
            stat = cudaSetDevice(i);
            stat = cudaMemGetInfo(&freemem, &totmem);
            if (stat != 0) {
            get_cuda_error(stat);
            cudaDevice = -1;
            }
            if (freemem > availmem) {
                availmem = freemem;
                cudaDevice = i;
            }
        }

    request = n_mat * sizeof(double)*max_dim*max_dim;

    if (request > availmem) cudaDevice = -1;
        createHandle();
    };
    
    void get_cuda_error(cudaError_t stat) {
        if (stat != cudaSuccess) {
            std::cerr << "CUDA Error: " << cudaGetErrorString(stat) << std::endl;
        }
    }

    void get_cublas_error(cublasStatus_t stat) {
        if (stat != CUBLAS_STATUS_SUCCESS) {
            std::cout << "CUBLAS Error: " << cublasGetStatusString(stat) << std::endl;
        }
    }

    void CudaRuntime::createHandle() {
        cublasStatus_t stat;
        cudaError_t stat_;
        stat_ = cudaSetDevice(cudaDevice);
        get_cuda_error(stat_);
        stat = cublasCreate(&handle);
        get_cublas_error(stat);
        stat = cublasGetVersion(handle, &version);
        get_cublas_error(stat);
    }

    void CudaRuntime::print_cuda_version(){
        std::cout << "Cuda version in use: " << std::to_string(version) << std::endl;
    };

} // namespace tgmctensor

