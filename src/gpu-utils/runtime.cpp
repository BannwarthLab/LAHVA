#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "runtime.hpp"
#include <iostream>
#include <string>

namespace tcgmtensor
{
    CudaRuntime::CudaRuntime(bool async_copy) 
    {  
        if (async_copy) this->enableAsyncCopy();
        createHandle();
    };

    CudaRuntime::~CudaRuntime() {
        if (stream_ != nullptr) get_cuda_error(cudaStreamDestroy(stream_));
        if (handle != nullptr) get_cublas_error(cublasDestroy(handle));
        //cudaDeviceReset();
    };

    size_t CudaRuntime::get_GPU_wmaxMem()
    {
        size_t freemem, totmem;
        int ndev;
        
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
        return availmem;
    }

    CudaRuntime::CudaRuntime(size_t max_dim, size_t n_mat) : 
    CudaRuntime( n_mat * sizeof(double)*max_dim*max_dim)
    {

    };

    CudaRuntime::CudaRuntime(size_t requestedMem)
    {
        size_t availmem = get_GPU_wmaxMem();

        if (requestedMem > availmem) cudaDevice = -1;
            createHandle();
    };    

    
    void get_cuda_ERROR(cudaError_t stat, const char* file, int line) {
        if (stat != cudaSuccess) {
            std::cerr << "CUDA Error: " << cudaGetErrorString(stat) << std::endl << "In File: "<< file << " at line: " << std::to_string(line)<< std::endl;
            throw std::runtime_error("CUDA Error");
        }
    }

    void get_cublas_ERROR(cublasStatus_t stat, const char* file, int line) {
        if (stat != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "CUBLAS Error: " << cublasGetStatusString(stat) << std::endl << "In File: "<< file << " at line: " << std::to_string(line)<< std::endl;
            throw std::runtime_error("CUBLAS Error");
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

    void CudaRuntime::enableAsyncCopy()
    {
        async_ = true;
        createStream();
    }

    void CudaRuntime::createStream()
    {
        get_cuda_error(cudaStreamCreateWithFlags(&stream_, streamFlag_));
    }  

    cusolverDnHandle_t CudaRuntime::getcuSolverHandle()
    {
        if (!this->cusolv_)
        {
            cusolv_ = std::make_shared<cuSolverRuntime>();
            if (stream_ == 0) this->createStream();
            cusolv_->setStream(this->getStream());
        }
        return this->cusolv_->getHandle();
    };
} // namespace tgmctensor

