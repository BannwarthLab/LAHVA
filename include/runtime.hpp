#ifndef TCGMBLAS_GPU_RUNTIME_HPP
#define TCGMBLAS_GPU_RUNTIME_HPP

#pragma warning(disable:2282)
#pragma warning(disable:815 858) 
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include "const.h"
#include <memory>
#include <iostream>
//#include <cublas_v2.h>
#include <cmath>
#define THREADS_PER_BLOCK 512
//Macro to get the line and file throwing cuda runtime errors
#define get_cuda_error(arg) get_cuda_ERROR(arg, __FILE__, __LINE__);
//Macro to get the line and file throwing cublas runtim errors
#define get_cublas_error(arg) get_cublas_ERROR(arg, __FILE__, __LINE__);

#define get_cusolv_error(arg) get_cusolv_ERROR(arg, __FILE__, __LINE__);

    /// @brief get error string for cuda runtime
    /// @param stat error ID
    /// @param file File in which error is raised
    /// @param line Line in which error is raised
    void get_cuda_ERROR(cudaError_t stat, const char * file, int line);
     /// @brief get error string for cublas runtime
    /// @param stat error ID
    /// @param file File in which error is raised
    /// @param line Line in which error is raised
    void get_cublas_ERROR(cublasStatus_t stat, const char* file, int line);

    /// @brief get error line and file of cusolver error
    /// @param stat error ID, an not be translated
    /// @param file file where error occurs
    /// @param line line where error occurs
    void get_cusolv_ERROR(cusolverStatus_t stat, const char *file, int line);

namespace tcgmtensor{
    
    

    class cuSolverRuntime;

    /// @brief cudaRuntime object, stream and Device and cublas Handle
    class CudaRuntime : public BLASRuntime {
    protected:
        /// @brief cudaDevice ID defaults to 1
        int cudaDevice = 0;
        /// @brief cuda Stream for asynchronous tasks
        cudaStream_t stream_ = 0;
        /// @brief flag to do memCopyAsync
        bool async_ = false;
        /// @brief store version number of cuda library
        int version = 0;
        /// @brief streamFlag used when creating stream
        unsigned int streamFlag_ = cudaStreamNonBlocking;
        /// @brief create cublasHandle
        void createHandle();
        size_t availMem_ = 0; 
        /// @brief blocksize used for launching kernels
        int blockSize_ = THREADS_PER_BLOCK;
        ///
        std::shared_ptr<cuSolverRuntime> cusolv_;
        bool delete_handle = false;
        bool delete_stream = false;
        bool critical_memory = false;
    public:

        /// @brief create Stream
        void createStream();
        /// @brief cublas Handle
        cublasHandle_t handle = nullptr;
        /// @brief default constructor not checking memory request
        CudaRuntime(bool async_copy = false);
        /// @brief Constructor checking memory request via tensor size
        /// @param max_dim maximum dimension of tensor
        /// @param n_mat number of amtrcies
        CudaRuntime(size_t max_dim, size_t n_mat, bool asnyc_copy = false);
        /// @brief Constructor checking memory request via number of bytes needed
        /// @param requestedMem hnumber of bytes needed for computation
        CudaRuntime(size_t requestedMem, bool async = false);
        /// @brief Destructor, destroying stream and handle if associated
        ~CudaRuntime();

        CudaRuntime(const CudaRuntime& other);
        CudaRuntime(CudaRuntime&& other);
        CudaRuntime& operator=(CudaRuntime&& other);
        CudaRuntime& operator=(const CudaRuntime& other);

        //%TODO move and copy Constructor
        ///
        cusolverDnHandle_t getcuSolverHandle();
        /// @brief check if runtime is setup for async task
        /// @return true if a stream is created
        bool asyncCopy() {return async_;};
        /// @brief check if runtime is setup for async task
        /// @return true if a stream is created
        bool asyncCopy() const {return async_;};
        /// @brief print cuda library version to stdout
        void print_cuda_version();
        /// @brief set internal async falg to true and createStream
        void enableAsyncCopy();
        
        /// @brief get CUDA Device ID
        /// @return CUDA device id
        inline int device_id() {return cudaDevice;};
        /// @brief get CUDA Device ID
        /// @return CUDA device id
        const inline int device_id() const {return cudaDevice;};

        void check_mem(size_t requestedMem);

        bool criticalSize(size_t ArrayMem) const {return ((double)((double)ArrayMem / (double)availMem_) > 0.025);}

        bool criticalMem() const {return critical_memory;};
        /// @brief Change block size
        /// @param blockSize new block size
        inline void setblockSize(int blockSize) {blockSize_ = blockSize;}
        /// @brief get blocksize
        /// @return blocksize for Kernel execution
        inline int blockSize() const {return blockSize_;};

        /// @brief determine gridSIze based on blocksize and Tensor dimensions
        /// @param base Tensor length as 1D
        /// @param exp dimension of Tensor
        /// @return grid Size for Kernel excution
        inline int gridSize(size_t base, size_t exp) const {return (int)ceil(((float)std::pow(base, exp)+blockSize_-1)/blockSize_);};
        
        /// @brief get device id of GPU with maximum working memory
        /// @return device ID of GPU with max memory
        size_t get_GPU_wmaxMem();
        
        /// @brief get async. stream
        /// @return CUDA stream
        cudaStream_t getStream() {return stream_;}
        cudaStream_t getStream() const {return stream_;}

        void cublasSetStream_() const 
        {
            
            get_cuda_error(cudaSetDevice(cudaDevice));
            get_cublas_error(cublasSetStream(handle, stream_));
        };
        
        /// @brief synchronize device after async. operations
        void synchronize() 
        {
            get_cuda_error(cudaStreamSynchronize(stream_));    
        }

        void synchronize() const
        {
            get_cuda_error(cudaStreamSynchronize(stream_));    
        }
    
    };

    class cuSolverRuntime
    {
    private:
        cusolverDnHandle_t handle = NULL;

    public:
        cuSolverRuntime();
        ~cuSolverRuntime();
        void setStream(const cudaStream_t& stream);
        cusolverDnHandle_t getHandle() { return handle; };
        cusolverDnHandle_t *getHandlePtr() { return &handle; }
    };
}
#endif