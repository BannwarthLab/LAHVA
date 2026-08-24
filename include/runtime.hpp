/// @file runtime.hpp
/// @brief CUDA runtime management including memory, streams, and error handling.
///
/// This header provides classes and utilities for managing CUDA device resources including
/// memory allocation, stream creation, and cuBLAS/cuSOLVER handle management. It includes
/// error reporting macros for convenient debugging and a CudaRuntime class that abstracts
/// CUDA device state and configuration for use in linear algebra operations.

#pragma once
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusolverDn.h>
#include <iostream>
#include <limits>
#include <memory>

#include "const.h"

/// @brief Default number of threads per block for CUDA kernel launches.
#define THREADS_PER_BLOCK 512

/// @brief Macro to capture CUDA runtime error with file and line information.
/// @param arg CUDA error status to check.
#define get_cuda_error(arg) get_cuda_ERROR(arg, __FILE__, __LINE__);

/// @brief Macro to capture cuBLAS error with file and line information.
/// @param arg cuBLAS error status to check.
#define get_cublas_error(arg) get_cublas_ERROR(arg, __FILE__, __LINE__);

/// @brief Macro to capture cuSOLVER error with file and line information.
/// @param arg cuSOLVER error status to check.
#define get_cusolv_error(arg) get_cusolv_ERROR(arg, __FILE__, __LINE__);

/// @brief Macro to capture cuSPARSE error with file and line information.
/// @param arg cuSPARSE error status to check.
#define get_cusparse_error(arg) get_cusparse_ERROR(arg, __FILE__, __LINE__);

/// @brief Print CUDA runtime error message with file and line information.
/// @param stat CUDA error status from a runtime call.
/// @param file Source file where the error occurred.
/// @param line Line number in the source file where the error occurred.
void get_cuda_ERROR(cudaError_t stat, const char * file, int line);

/// @brief Print cuBLAS error message with file and line information.
/// @param stat cuBLAS error status from a BLAS operation.
/// @param file Source file where the error occurred.
/// @param line Line number in the source file where the error occurred.
void get_cublas_ERROR(cublasStatus_t stat, const char* file, int line);

/// @brief Print cuSOLVER error message with file and line information.
/// @param stat cuSOLVER error status from a solver operation.
/// @param file Source file where the error occurred.
/// @param line Line number in the source file where the error occurred.
void get_cusolv_ERROR(cusolverStatus_t stat, const char *file, int line);

/// @brief Print cuSPARSE error message with file and line information.
/// @param stat cuSPARSE error status from a sparse operation.
/// @param file Source file where the error occurred.
/// @param line Line number in the source file where the error occurred.
void get_cusparse_ERROR(cusparseStatus_t stat, const char* file, int line);

namespace lahva{
    
    
    // Forward declaration
    class cuSolverRuntime;
    class cuSparseRuntime;

    /// @brief GPU runtime for CUDA-accelerated BLAS and LAPACK operations.
    ///
    /// Manages CUDA device resources including memory allocation, stream creation,
    /// and cuBLAS/cuSOLVER handle lifecycle. Provides a unified interface for GPU
    /// computation with automatic memory checking, async copy support, and kernel
    /// configuration (block size and grid size calculation).
    class CudaRuntime : public BLASRuntime {
    protected:
        /// @brief CUDA device ID (default: -1 for no device).
        /// Set automatically by device selection or explicitly during initialization.
        int cudaDevice = -1; 
        
        /// @brief Flag enabling asynchronous memory copies and kernel launches.
        /// When true, operations use the runtime's CUDA stream for non-blocking execution.
        bool async_ = true;
        
        /// @brief CUDA driver version number.
        /// Populated by initialization code to track installed CUDA version.
        int version = 0;
        
        /// @brief Stream creation flags (e.g., cudaStreamNonBlocking).
        /// Controls behavior of CUDA streams created by createStream().
        unsigned int streamFlag_ = cudaStreamNonBlocking;
        
        /// @brief Create and initialize the cuBLAS handle.
        void createHandle();
        
        /// @brief Available GPU memory in bytes.
        size_t availMem_ = 0; 
        
        /// @brief Block size for CUDA kernel launches (default: THREADS_PER_BLOCK).
        mutable int blockSize_ = THREADS_PER_BLOCK;
        
        /// @brief Maximum allowed block dimension for kernel grid calculation.
        mutable size_t max_block_ = std::numeric_limits<unsigned int>::max(); 
        
        /// @brief Shared pointer to cuSOLVER runtime for dense solver operations.
        /// Lazily initialized on first access via getcuSolverHandle().
        std::shared_ptr<cuSolverRuntime> cusolv_;
        
        /// @brief Shared pointer to cuSPARSE runtime for sparse matrix operations.
        /// Lazily initialized on first access via getcuSparseHandle().
        std::shared_ptr<cuSparseRuntime> cusparse_;
        
        /// @brief Flag indicating whether the cuBLAS handle should be deleted by destructor.
        bool delete_handle = false;
        
        /// @brief Flag indicating whether the CUDA stream should be deleted by destructor.
        bool delete_stream = false;
        
        /// @brief Flag indicating critical memory conditions (threshold exceeded).
        /// Set when available memory drops below critical_memory_threshold.
        bool critical_memory = false;
        
        /// @brief Memory usage threshold ratio for critical memory detection (default: 0.6).
        /// When available memory ratio drops below this, critical_memory flag is set.
        float critical_memory_threshold = 0.6;
        
        /// @brief Shared pointer to the CUDA stream for async operations.
        /// Defaults to the per-thread stream; can be overridden in derived classes.
        mutable std::shared_ptr<cudaStream_t> stream_ = std::make_shared<cudaStream_t>(cudaStreamPerThread);
        
    public:

        /// @brief Create and initialize a CUDA stream with streamFlag_ configuration.
        void createStream();
        
        /// @brief cuBLAS handle for GPU BLAS operations (nullptr if not initialized).
        /// Automatically created by constructors; accessible to operations needing direct cuBLAS access.
        cublasHandle_t handle = nullptr;
        
        /// @brief Default constructor creating runtime without memory pre-allocation.
        /// @param async_copy If true, enable asynchronous memory copies and kernel execution.
        CudaRuntime(bool async_copy = false);
        
        /// @brief Constructor with memory pre-check based on tensor dimensions.
        /// @param max_dim Maximum dimension of tensors to allocate (square tensors of size max_dim × max_dim).
        /// @param n_mat Number of matrices to allocate space for.
        /// @param asnyc_copy If true, enable asynchronous memory copies and kernel execution.
        /// @throws std::runtime_error If insufficient GPU memory is available for the requested allocation.
        CudaRuntime(size_t max_dim, size_t n_mat, bool asnyc_copy = false);
        
        /// @brief Constructor with memory pre-check based on byte requirement.
        /// @param requestedMem Total number of bytes needed for GPU computation.
        /// @param async If true, enable asynchronous memory copies and kernel execution.
        /// @throws std::runtime_error If insufficient GPU memory is available for the requested allocation.
        CudaRuntime(size_t requestedMem, bool async = false);
        
        /// @brief Destructor releasing GPU resources (stream and handle if owned).
        /// Synchronizes GPU before cleanup to ensure all operations complete.
        ~CudaRuntime();

        /// @brief Copy constructor.
        /// @param other Source runtime to copy from.
        CudaRuntime(const CudaRuntime& other);
        
        /// @brief Move constructor.
        /// @param other Source runtime to move from.
        CudaRuntime(CudaRuntime&& other);
        
        /// @brief Move assignment operator.
        /// @param other Source runtime to move from.
        /// @return Reference to this runtime.
        CudaRuntime& operator=(CudaRuntime&& other);
        
        /// @brief Copy assignment operator.
        /// @param other Source runtime to copy from.
        /// @return Reference to this runtime.
        CudaRuntime& operator=(const CudaRuntime& other);

        /// @brief Retrieve or create the cuSOLVER dense handle.
        /// @return cusolverDnHandle_t for GPU linear algebra solver operations.
        cusolverDnHandle_t getcuSolverHandle();

        /// @brief Retrieve or create the cuSPARSE handle.
        /// @return cusparseHandle_t for GPU sparse matrix operations.
        cusparseHandle_t getcuSparseHandle();

        /// @brief Check if runtime is configured for asynchronous operations.
        /// @return True if async copy and stream-based execution are enabled.
        bool asyncCopy() {return async_;};
        
        /// @brief Check if runtime is configured for asynchronous operations (const version).
        /// @return True if async copy and stream-based execution are enabled.
        bool asyncCopy() const {return async_;};
        
        /// @brief Print CUDA driver and runtime version information to standard output.
        void print_cuda_version();
        
        /// @brief Enable asynchronous memory copies and kernel execution.
        /// Creates a CUDA stream if not already created and sets async_ flag to true.
        void enableAsyncCopy();
        
        /// @brief Set the threshold ratio for critical memory condition detection.
        /// @param ratio Memory ratio threshold (e.g., 0.6 = 60% usage triggers critical condition).
        void setCriticalMemoryRatio(float ratio) {critical_memory_threshold = ratio;};
        
        /// @brief Retrieve the CUDA device ID managed by this runtime.
        /// @return CUDA device ID (-1 if no device is selected).
        inline int device_id() {return cudaDevice;};
        
        /// @brief Retrieve the CUDA device ID managed by this runtime (const version).
        /// @return CUDA device ID (-1 if no device is selected).
        inline int device_id() const {return cudaDevice;};

        /// @brief Check if requested memory is available on GPU, reconfigure if needed.
        /// @param requestedMem Number of bytes required for computation.
        /// @param force_new_handle If true, force recreation of cuBLAS handle even if memory is available.
        void check_mem(size_t requestedMem, bool force_new_handle = false);

        /// @brief Check if a tensor allocation size is critical (potentially problematic).
        /// Critical size threshold is 2.5% of available GPU memory.
        /// @param ArrayMem Number of bytes for the tensor allocation.
        /// @return True if the allocation exceeds the critical size threshold.
        bool criticalSize(size_t ArrayMem) const {return ((double)((double)ArrayMem / (double)availMem_) > 0.025);}

        /// @brief Check if GPU memory is in critical condition.
        /// @return True if memory usage has exceeded critical_memory_threshold.
        bool criticalMem() const {return critical_memory;};
        
        /// @brief Set the block size for CUDA kernel launches.
        /// @param blockSize Number of threads per block (should not exceed device limits).
        inline void setblockSize(int blockSize) {blockSize_ = blockSize;}
        
        /// @brief Set the block size for CUDA kernel launches (const version for mutable modification).
        /// @param blockSize Number of threads per block (should not exceed device limits).
        inline void setblockSize(int blockSize) const {blockSize_ = blockSize;}
        
        /// @brief Retrieve the current block size for kernel launches.
        /// @return Number of threads per block.
        inline int blockSize() const {return blockSize_;};

        /// @brief Calculate grid size (number of blocks) for kernel execution.
        /// Computes optimal grid dimension based on block size and total work (base^exp).
        /// @param base Base dimension of tensor (e.g., matrix dimension).
        /// @param exp Exponent for tensor (power to which base is raised for total elements).
        /// @return Number of blocks to launch (clamped by max_block_).
        inline size_t gridSize(size_t base, size_t exp) const {return (size_t)std::min(ceil(((float)std::pow(base, exp))/blockSize_),(float)this->max_block_);};
        
        /// @brief Find and select the GPU device with maximum available memory.
        /// Automatically sets cudaDevice to the selected GPU.
        /// @return Device ID of the GPU with maximum available memory.
        size_t get_GPU_wmaxMem();
        
        /// @brief Retrieve a reference to the CUDA stream for async operations.
        /// @return Reference to the current CUDA stream.
        cudaStream_t& getStream() {return *(stream_);}
        
        /// @brief Retrieve a reference to the CUDA stream for async operations (const version).
        /// @return Const reference to the current CUDA stream.
        cudaStream_t& getStream() const {return *(stream_);}

        /// @brief Retrieve a shared pointer to the CUDA stream.
        /// @return Shared pointer to the stream for external stream management.
        std::shared_ptr<cudaStream_t>& getStreamPtr() {return stream_;}
        
        /// @brief Retrieve a shared pointer to the CUDA stream (const version).
        /// @return Const reference to shared pointer to the stream.
        std::shared_ptr<cudaStream_t>& getStreamPtr() const {return stream_;}

        /// @brief Copy the stream pointer to an external shared pointer.
        /// @param stream Shared pointer to receive this runtime's stream.
        void getStream_ptr(std::shared_ptr<cudaStream_t>& stream) {stream = stream_;}
        
        /// @brief Copy the stream pointer to an external shared pointer (const version).
        /// @param stream Shared pointer to receive this runtime's stream.
        void getStream_ptr(std::shared_ptr<cudaStream_t>& stream) const {stream = stream_;}

        /// @brief Associate the cuBLAS handle with the runtime's CUDA stream.
        /// Sets the device and attaches the stream to the handle for subsequent operations.
        inline void cublasSetStream_() const 
        {
            
            get_cuda_error(cudaSetDevice(cudaDevice));
            get_cublas_error(cublasSetStream(handle, *stream_));
        };
        
        /// @brief Synchronize the CUDA stream to wait for all pending operations.
        /// Blocks until all kernels and memory operations on this runtime's stream complete.
        void synchronize() 
        {
            get_cuda_error(cudaStreamSynchronize(*stream_));    
        }

        /// @brief Synchronize the CUDA stream to wait for all pending operations (const version).
        /// Blocks until all kernels and memory operations on this runtime's stream complete.
        void synchronize() const
        {
            get_cuda_error(cudaStreamSynchronize(*stream_));    
        }
    
    };

    /// @brief Wrapper class for cuSOLVER dense solver handle and stream management.
    ///
    /// Manages the lifetime of a cusolverDn handle for dense linear algebra
    /// operations on GPU. Provides methods to set the CUDA stream and access
    /// the underlying handle and handle pointer.
    class cuSolverRuntime
    {
    private:
        /// @brief Dense solver handle for GPU linear algebra operations.
        cusolverDnHandle_t handle = NULL;

    public:
        /// @brief Constructor initializing the cuSOLVER handle.
        cuSolverRuntime();
        
        /// @brief Destructor releasing the cuSOLVER handle.
        ~cuSolverRuntime();
        
        /// @brief Associate the solver handle with a specific CUDA stream.
        /// @param stream CUDA stream to use for all solver operations.
        void setStream(const cudaStream_t& stream);
        
        /// @brief Retrieve the cuSOLVER dense handle.
        /// @return The cusolverDnHandle_t managed by this runtime.
        cusolverDnHandle_t getHandle() { return handle; };
        
        /// @brief Retrieve a pointer to the cuSOLVER dense handle.
        /// @return Pointer to the cusolverDnHandle_t managed by this runtime.
        cusolverDnHandle_t *getHandlePtr() { return &handle; }
    };

    /// @brief GPU cuSPARSE runtime for sparse matrix operations.
    ///
    /// Manages cuSPARSE handle lifecycle and provides stream management for
    /// GPU-accelerated sparse linear algebra operations.
    class cuSparseRuntime
    {
    private:
        /// @brief Sparse matrix handle for GPU sparse linear algebra operations.
        cusparseHandle_t handle = NULL;

    public:
        /// @brief Constructor initializing the cuSPARSE handle.
        cuSparseRuntime();

        /// @brief Destructor releasing the cuSPARSE handle.
        ~cuSparseRuntime();

        /// @brief Associate the sparse handle with a specific CUDA stream.
        /// @param stream CUDA stream to use for all sparse operations.
        void setStream(const cudaStream_t& stream);

        /// @brief Retrieve the cuSPARSE handle.
        /// @return The cusparseHandle_t managed by this runtime.
        cusparseHandle_t getHandle() { return handle; };

        /// @brief Retrieve a pointer to the cuSPARSE handle.
        /// @return Pointer to the cusparseHandle_t managed by this runtime.
        cusparseHandle_t *getHandlePtr() { return &handle; }
    };
}