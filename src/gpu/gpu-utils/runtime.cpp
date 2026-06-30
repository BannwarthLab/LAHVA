/// @file runtime.cpp
/// @brief CUDA and cuBLAS runtime management implementation.
///
/// Provides CUDA device initialization, memory management, and cuBLAS context setup
/// for GPU-accelerated linear algebra operations.

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <omp.h>
#include <string>
#include "runtime.hpp"

/// @brief Checks CUDA error status and throws on failure.
///
/// Validates CUDA API return codes and throws an exception with file/line information on error.
///
/// @param stat CUDA error code to check.
/// @param file Source file where error check occurred.
/// @param line Line number where error check occurred.
void get_cuda_ERROR(cudaError_t stat, const char* file, int line) {
        if (stat != cudaSuccess) {
            std::cerr << "CUDA Error: " << cudaGetErrorString(stat) << std::endl << "In File: "<< file << " at line: " << std::to_string(line)<< std::endl;
            throw std::runtime_error("CUDA Error");
        }
    }

    /// @brief Checks cuBLAS error status and throws on failure.
    ///
    /// Validates cuBLAS API return codes and throws an exception with file/line information on error.
    ///
    /// @param stat cuBLAS status code to check.
    /// @param file Source file where error check occurred.
    /// @param line Line number where error check occurred.
    void get_cublas_ERROR(cublasStatus_t stat, const char* file, int line) {
        if (stat != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "CUBLAS Error: " << cublasGetStatusString(stat) << std::endl << "In File: "<< file << " at line: " << std::to_string(line)<< std::endl;
            throw std::runtime_error("CUBLAS Error");
        }
    }
namespace lahva
{
    /// @brief Initializes CUDA runtime with device and cuBLAS setup.
    ///
    /// Selects GPU device with maximum available memory and creates cuBLAS handle
    /// for GPU-accelerated linear algebra operations.
    ///
    /// @param async_copy Whether to enable asynchronous memory copying (default: false).
    CudaRuntime::CudaRuntime(bool async_copy)
    {
        if (async_copy) this->enableAsyncCopy();
        get_GPU_wmaxMem();
        createHandle();
    };

    /// @brief Destroys CUDA runtime and releases GPU resources.
    ///
    /// Cleans up CUDA streams and cuBLAS handles created during initialization.
    CudaRuntime::~CudaRuntime() {
        if (delete_stream) get_cuda_error(cudaStreamDestroy(*stream_));
        if (delete_handle) 
        {
            get_cublas_error(cublasDestroy(handle));
        }
        //cudaDeviceReset();
    };

    /// @brief Selects GPU device with maximum available memory.
    ///
    /// Queries all available GPU devices and selects the one with the most free memory.
    /// In multi-threaded contexts, distributes threads across multiple GPUs for load balancing.
    ///
    /// @return Amount of free memory on selected device in bytes.
    size_t CudaRuntime::get_GPU_wmaxMem()
    {
        int thread = omp_get_thread_num();
        int level = omp_get_active_level();
        
        size_t freemem, totmem;
        int ndev;
        
        size_t availmem = 0;

        freemem = 0;
        totmem = 0;
	    get_cuda_error(cudaGetDeviceCount(&ndev));
        if( ndev > 1 and level > 0)
        {
            int i = thread % ndev;
            get_cuda_error(cudaSetDevice(i));
            get_cuda_error(cudaMemGetInfo(&freemem, &totmem));
            if (freemem > availmem) {
                availmem = freemem;
                cudaDevice = i;
            }
        } 
        else
        {
        for (int i = 0; i < ndev; ++i) {
            get_cuda_error(cudaSetDevice(i));
            get_cuda_error(cudaMemGetInfo(&freemem, &totmem));
            if (freemem > availmem) {
                availmem = freemem;
                cudaDevice = i;
            }
        }
    }
        return availmem;
    }

    /// @brief Initializes CUDA runtime with memory allocation sizing.
    ///
    /// Allocates GPU memory and optimizes block size for matrix operations of given dimensions.
    /// Selects block size (512, 256, or 128 threads) to best distribute work across GPU SMs.
    ///
    /// @param max_dim Maximum matrix dimension for memory sizing.
    /// @param n_mat Number of matrices to allocate space for.
    /// @param async Whether to enable asynchronous memory copying.
    CudaRuntime::CudaRuntime(size_t max_dim, size_t n_mat, bool async) :
    CudaRuntime( n_mat * 8*max_dim*max_dim, async)
    {
        if (cudaDevice ==-1) {
            return;
        }
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, this->cudaDevice);
        size_t n_sm = deviceProp.multiProcessorCount;

        size_t best_block_size = 0;
        int rest = -500;

        for (auto i : { 512,  256, 128})
        {
            this->blockSize_ = i;
            size_t gridS = this->gridSize(max_dim*max_dim, 1);
            if (gridS % n_sm < rest)
            {
                best_block_size = i;
                rest = gridS % n_sm;
                if (rest == 0) continue;
            }
        }
        this->blockSize_ = 512;
    };

    /// @brief Initializes CUDA runtime with requested memory allocation.
    ///
    /// Allocates specified amount of GPU memory and initializes cuBLAS handle.
    ///
    /// @param requestedMem Amount of GPU memory to allocate in bytes.
    /// @param async Whether to enable asynchronous memory copying.
    CudaRuntime::CudaRuntime(size_t requestedMem, bool async)
    {
        if (async) this->enableAsyncCopy();
        check_mem(requestedMem, true);
    };

    /// @brief Move constructor - transfers GPU resources from another runtime.
    ///
    /// Transfers ownership of CUDA device, streams, and cuBLAS handle from source to this instance.
    /// Source object relinquishes ownership to prevent double-deletion.
    ///
    /// @param other CudaRuntime instance to move from (moved-from object becomes invalid).
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

    /// @brief Validates available GPU memory and initializes cuBLAS handle.
    ///
    /// Checks that requested memory is available on GPU device. Sets critical_memory flag
    /// if allocation would exceed threshold. Creates or updates cuBLAS handle as needed.
    ///
    /// @param requestedMem Amount of memory required in bytes.
    /// @param force_new_handle Force creation of new cuBLAS handle even if one exists.
    void CudaRuntime::check_mem(size_t requestedMem, bool force_new_handle)
    {
        if (cudaDevice == -1)
        get_GPU_wmaxMem();

        size_t totmem, freemem;
        get_cuda_error(cudaMemGetInfo(&freemem, &totmem));
        availMem_ = freemem;
        if (requestedMem > availMem_)
        {
            cudaDevice = -1;
            return;
        }

        if ((float)((float)requestedMem / (float) availMem_) > critical_memory_threshold)
        {
            critical_memory = true;
        }

        if ((handle == nullptr) or force_new_handle)
        {
            createHandle();
        }
    }

    /// @brief Copy constructor - creates new independent CUDA runtime.
    ///
    /// Constructs new CudaRuntime with same async settings as source,
    /// acquiring separate GPU resources and cuBLAS handle.
    ///
    /// @param other CudaRuntime instance to copy async settings from.
    CudaRuntime::CudaRuntime(const CudaRuntime& other) : CudaRuntime(other.async_)
    {
    }

    /// @brief Copy assignment operator - replaces GPU resources.
    ///
    /// Transfers async settings from source and acquires new GPU device and cuBLAS handle.
    ///
    /// @param other CudaRuntime to copy from.
    /// @return Reference to this object.
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

    /// @brief Move assignment operator - transfers GPU resources.
    ///
    /// Transfers ownership of CUDA device, streams, and cuBLAS handle from source.
    /// Source object relinquishes ownership to prevent double-deletion.
    ///
    /// @param other CudaRuntime to move from (becomes invalid after move).
    /// @return Reference to this object.
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

    /// @brief Creates or recreates cuBLAS handle for current GPU device.
    ///
    /// Destroys existing handle if present, sets current CUDA device,
    /// creates new cuBLAS handle, and retrieves cuBLAS version information.
    void CudaRuntime::createHandle()
    {
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

    /// @brief Prints CUDA device properties and cuBLAS version to stdout.
    ///
    /// Displays async engine count, concurrent kernel capability,
    /// and cuBLAS version running on the selected GPU device.
    void CudaRuntime::print_cuda_version()
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, cudaDevice);
        std::cout << "Async engine count: " << prop.asyncEngineCount << std::endl;
        std::cout << "Concurrent kernels: " << prop.concurrentKernels << std::endl;
        std::cout << "Cuda version in use: " << std::to_string(version) << "on device: " << cudaDevice << std::endl;
    };

    /// @brief Enables asynchronous memory copying mode.
    ///
    /// Sets async flag to true, allowing concurrent CPU-GPU data transfers
    /// and kernel execution for better performance.
    void CudaRuntime::enableAsyncCopy()
    {
        async_ = true;
    }

    /// @brief Creates CUDA stream for asynchronous operations.
    ///
    /// Destroys existing stream if present, then creates new stream with
    /// appropriate flags for non-blocking GPU operations.
    void CudaRuntime::createStream()
    {
        if (delete_stream) get_cuda_error(cudaStreamDestroy(*stream_));
        get_cuda_error(cudaStreamCreateWithFlags(stream_.get(), streamFlag_));
        delete_stream = true;
    }

    /// @brief Gets or creates cuSolver handle for linear solver operations.
    ///
    /// Lazily constructs cuSolver runtime on first call, associating it with
    /// the CUDA stream from this runtime for proper synchronization.
    ///
    /// @return cuSolver handle for dense matrix operations.
    cusolverDnHandle_t CudaRuntime::getcuSolverHandle()
    {
        if (!this->cusolv_)
        {
            cusolv_ = std::make_shared<cuSolverRuntime>();
            cusolv_->setStream(this->getStream());
        }
        return this->cusolv_->getHandle();
    };
} // namespace lahva
