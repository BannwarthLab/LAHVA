#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "runtime.hpp"
#include <iostream>
#include <string>


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
namespace lahva
{
    CudaRuntime::CudaRuntime(bool async_copy) 
    {  
        if (async_copy) this->enableAsyncCopy();
        get_GPU_wmaxMem();
        createHandle();
    };

    CudaRuntime::~CudaRuntime() {
        if (delete_stream) get_cuda_error(cudaStreamDestroy(stream_));
        if (delete_handle) 
        {
            get_cublas_error(cublasDestroy(handle));
        }
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

    CudaRuntime::CudaRuntime(size_t max_dim, size_t n_mat, bool async) : 
    CudaRuntime( n_mat * 8*max_dim*max_dim, async)
    {
        
    };

    CudaRuntime::CudaRuntime(size_t requestedMem, bool async)
    {
        if (async) this->enableAsyncCopy();
        check_mem(requestedMem);   
    };    

    CudaRuntime::CudaRuntime(CudaRuntime&& other) 
    {
        cudaDevice = other.cudaDevice;
        async_ = other.async_;
        version = other.version;
        handle = other.handle;
        stream_ = other.stream_;

        other.delete_handle = false;
        other.delete_stream = false;
        
    }

    void CudaRuntime::check_mem(size_t requestedMem) 
    {
        availMem_ = get_GPU_wmaxMem();
        if (requestedMem > availMem_) 
        {
            cudaDevice = -1;
            return;
        }
        //std::cout <<  "requested: " << requestedMem/std::pow(1024,3)  << "GB available: " << availMem_ / std::pow(1024,3) << " GB" << std::endl;    
        if ((double)((double)requestedMem / (double)availMem_) > 0.60)
        {
            critical_memory = true;
        }

        createHandle();
    }
    
    
    CudaRuntime::CudaRuntime(const CudaRuntime& other) : CudaRuntime(other.async_)
    {
        
    }

    CudaRuntime& CudaRuntime::operator=(const CudaRuntime& other) 
    {
        if (this != &other)
        {
            this->async_ = other.async_;
            if (this->async_) this->enableAsyncCopy();
            this->get_GPU_wmaxMem();
            this->createHandle();
        }
        return *this;
        
    }

    CudaRuntime& CudaRuntime::operator=(CudaRuntime&& other)
    {
        if (this != &other)
        {
            cudaDevice = other.cudaDevice;
            async_ = other.async_;
            version = other.version;
            handle = other.handle;
            stream_ = other.stream_;
            delete_handle = other.delete_handle;
            delete_stream = other.delete_stream;

            other.delete_handle = false;
            other.delete_stream = false;
            
        }
        return *this;
    }

    void CudaRuntime::createHandle() {
        cublasStatus_t stat;
        cudaError_t stat_;
        if (delete_handle) get_cublas_error(cublasDestroy(handle));
        
        
        stat_ = cudaSetDevice(cudaDevice);
        get_cuda_error(stat_); 
        stat = cublasCreate(&handle);
        get_cublas_error(stat);
        stat = cublasGetVersion(handle, &version);
        get_cublas_error(stat);
        delete_handle = true;
        
    }

    void CudaRuntime::print_cuda_version(){
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, cudaDevice);
        std::cout << "Async engine count: " << prop.asyncEngineCount << std::endl;
        std::cout << "Concurrent kernels: " << prop.concurrentKernels << std::endl;
        std::cout << "Cuda version in use: " << std::to_string(version) << "on device: "<< cudaDevice << std::endl;
    };

    void CudaRuntime::enableAsyncCopy()
    {
        async_ = true;
        stream_ = cudaStreamPerThread;
        //createStream();
    }

    void CudaRuntime::createStream()
    {
        get_cuda_error(cudaStreamCreateWithFlags(&stream_, streamFlag_));
        delete_stream = true;
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

