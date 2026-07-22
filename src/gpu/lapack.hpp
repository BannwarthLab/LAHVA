/// @file lapack.hpp
/// @brief GPU-accelerated LAPACK operations for solving linear systems.
///
/// Provides GPU kernel implementations for LAPACK-style linear system solvers
/// using CUDA's cuSolver library. Supports solving general systems of linear equations
/// with LU factorization. Each operation is provided for both double and float precision
/// with GPU runtime management.

#include "impl/gpu/utils.hpp"
#include "linalg.hpp"
#include "runtime.hpp"

namespace lahva
{
    namespace gpu
    {
        template<typename T, class Allocator, class GPUAllocator>
        void SolveGenSysLinEquations(CudaRuntime& cudart, const char* Ta, Matrix<T, Allocator, GPUAllocator>& a, Matrix<T, Allocator, GPUAllocator>& b)
        {

            check_device_alloc(cudart, a);
            check_device_alloc(cudart, b);
            cudaDataType data = CUDA_R_64F; 
            using MyVector = Vector<int64_t> ;
            if ( a.shape().first != a.shape().second)
            {
                throw std::runtime_error("A should be symmetric");
            };
            int64_t n = a.shape().first;
            int64_t nrhs = b.shape().second;
            cusolverDnParams_t params;
            size_t wspacedevice;
            size_t wspacehost;
            get_cusolv_error(cusolverDnCreateParams(&params));
            get_cusolv_error(cusolverDnXgetrf_bufferSize(cudart.getcuSolverHandle(), params, n, n, data, a.gpu_data(), n, data, &wspacedevice, &wspacehost));
            
            if (b.shape().first != n)
            {
                throw std::runtime_error("B should have as much rows as A. B(n, nrhs)");
            }
            MyVector ipiv(n, a.get_allocator(), a.get_gpuallocator());
            ipiv.copy2device(cudart);
            void * buffer_device;
            get_cuda_error(cudaMalloc(&buffer_device, wspacedevice));
            void * buffer_host = malloc(wspacehost);
            MyVector info(1, 0, a.get_allocator(), a.get_gpuallocator());
            info.copy2device(cudart);

            get_cusolv_error(cusolverDnXgetrf(cudart.getcuSolverHandle(), params, n, n, data, a.gpu_data(), n, ipiv.gpu_data(),
                                               data, buffer_device, wspacedevice, buffer_host, wspacehost, info.gpu_data()));
            
            get_cuda_error(cudaFree(buffer_device));
            free(buffer_host);

            info.copy2host(cudart);
            if (info.sum() != 0)
            {
                throw std::runtime_error("Failure in DGETRF");
            }
            cublasOperation_t transa = get_trans(Ta);
            info.copy2device(cudart);
            get_cusolv_error(cusolverDnXgetrs(cudart.getcuSolverHandle(), params, transa, n, nrhs, data, a.gpu_data(), n, ipiv.gpu_data(),
                                               data, b.gpu_data(), n, info.gpu_data()));
            info.copy2host(cudart);
            if (info.sum() != 0)
            {
                throw std::runtime_error("Failure in DGETRS");
            }
            
        };

        template<typename T, typename Allocator, typename GPUAllocator>
        void SolveGenSysLinEquations(CudaRuntime& cudart, Matrix<T, Allocator, GPUAllocator>& a, Matrix<T, Allocator, GPUAllocator>& b, const char* Ta)
        {

            check_device_alloc(cudart, a);
            check_device_alloc(cudart, b);

            cudaDataType data = CUDA_R_64F; 

            if ( a.shape().first != a.shape().second)
            {
                throw std::runtime_error("A should be symmetric");
            };
            int64_t n = a.shape().first;
            int64_t nrhs = b.shape().second;
            cusolverDnParams_t params;
            size_t wspacedevice;
            size_t wspacehost;
            get_cusolv_error(cusolverDnCreateParams(&params));
            get_cusolv_error(cusolverDnXgetrf_bufferSize(cudart.getcuSolverHandle(), params, n, n, data, a.gpu_data(), n, data, &wspacedevice, &wspacehost));
            
            if (b.shape().first != n)
            {
                throw std::runtime_error("B should have as much rows as A. B(n, nrhs)");
            }
            Vector<int64_t> ipiv(n, a.get_allocator(), a.get_gpuallocator());
            ipiv.copy2device(cudart);
            void * buffer_device;
            get_cuda_error(cudaMalloc(&buffer_device, wspacedevice));
            void * buffer_host = malloc(wspacehost);
            Vector<int64_t> info(1, 0, a.get_allocator(), a.get_gpuallocator());
            info.copy2device(cudart);

            get_cusolv_error(cusolverDnXgetrf(cudart.getcuSolverHandle(), params, n, n, data, a.gpu_data(), n, ipiv.gpu_data(),
                                               data, buffer_device, wspacedevice, buffer_host, wspacehost, info.gpu_data()));
            
            get_cuda_error(cudaFree(buffer_device));
            free(buffer_host);

            info.copy2host(cudart);
            if (info.sum() != 0)
            {
                throw std::runtime_error("Failure in DGETRF");
            }
            cublasOperation_t transa = get_trans(Ta);
            info.copy2device(cudart);
            get_cusolv_error(cusolverDnXgetrs(cudart.getcuSolverHandle(), params, transa, n, nrhs, data, a.gpu_data(), n, ipiv.gpu_data(),
                                               data, b.gpu_data(), n, info.gpu_data()));
            info.copy2host(cudart);
            if (info.sum() != 0)
            {
                throw std::runtime_error("Failure in DGETRS");
            }
            
        };

    }
    
} // namespace lahva
