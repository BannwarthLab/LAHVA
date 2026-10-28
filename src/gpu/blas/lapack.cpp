/// @file lapack.cpp
/// @brief GPU-accelerated LAPACK operations implementation.
///
/// Implements GPU versions of LAPACK-style operations using NVIDIA's cuSOLVER dense library.
/// Currently supports the inversion of general square matrices and solving general systems of linear equations.

#include "impl/gpu/utils.hpp"
#include "impl/blas/gpu/lapack.hpp"
#include "linalg.hpp"
#include "runtime.hpp"
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace lahva
{
    namespace gpu
    {
        /// @brief Computes the inverse of a general square matrix (DGETRF + DGETRS).
        ///
        /// Computes the LU factorization with partial pivoting of A and solves A*X = I, so that X = A^(-1).
        /// The inverse is stored back in A on the device.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param a Input/output square matrix, replaced with its inverse.
        template <typename T>
        void InvertMatrix(CudaRuntime &cudart, Matrix_<T> &a)
        {
            if (a.shape().first != a.shape().second)
            {
                throw std::runtime_error("A should be square");
            }
            check_device_alloc(cudart, a);

            int64_t n = static_cast<int64_t>(a.shape().first);
            cusolverDnHandle_t handle = cudart.getcuSolverHandle();
            cudaStream_t stream = cudart.getStream();
            cudaDataType data = std::is_same_v<T, double> ? CUDA_R_64F : CUDA_R_32F;

            cusolverDnParams_t raw_params;
            get_cusolv_error(cusolverDnCreateParams(&raw_params));
            std::unique_ptr<cusolverDnParams, decltype(&cusolverDnDestroyParams)> params(raw_params, &cusolverDnDestroyParams);

            // Right-hand side B = I (built on the device only), overwritten with A^(-1) by getrs
            Matrix<T> inv(Shape(n, n), cudart);
            check_device_alloc(cudart, inv);
            get_cuda_error(cudaMemsetAsync(inv.gpu_data(), 0, sizeof(T) * n * n, stream));
            Vector<T> ones(n, (T)1.0);
            inv.set_diagonal(cudart, ones);

            size_t wspacedevice = 0;
            size_t wspacehost = 0;
            get_cusolv_error(cusolverDnXgetrf_bufferSize(handle, params.get(), n, n, data, a.gpu_data(), n, data,
                                                         &wspacedevice, &wspacehost));

            // Device-only buffers: pivot indices and getrf workspace
            Vector<int64_t> ipiv(n, cudart);
            check_device_alloc(cudart, ipiv);
            Vector<T> work_device((wspacedevice + sizeof(T) - 1) / sizeof(T), cudart);
            check_device_alloc(cudart, work_device);
            std::vector<char> work_host(wspacehost);
            Vector<int> info(1, 0);
            info.copy2device(cudart);

            get_cusolv_error(cusolverDnXgetrf(handle, params.get(), n, n, data, a.gpu_data(), n, ipiv.gpu_data(), data,
                                              work_device.gpu_data(), wspacedevice, work_host.data(), wspacehost,
                                              info.gpu_data()));

            info.copy2host(cudart);
            if (info.sum() != 0)
            {
                throw std::runtime_error("Failure in GETRF");
            }
            info.copy2device(cudart);
            get_cusolv_error(cusolverDnXgetrs(handle, params.get(), CUBLAS_OP_N, n, n, data, a.gpu_data(), n, ipiv.gpu_data(),
                                              data, inv.gpu_data(), n, info.gpu_data()));
            get_cuda_error(cudaMemcpyAsync(a.gpu_data(), inv.gpu_data(), sizeof(T) * n * n, cudaMemcpyDeviceToDevice, stream));
            info.copy2host(cudart);
            if (info.sum() != 0)
            {
                throw std::runtime_error("Failure in GETRS");
            }
        }

        template void InvertMatrix<double>(CudaRuntime &cudart, Matrix_<double> &a);
        template void InvertMatrix<float>(CudaRuntime &cudart, Matrix_<float> &a);

        /// @brief Solves a general system of linear equations op(A)*X = B (GETRF + GETRS).
        ///
        /// Computes the LU factorization with partial pivoting of A and solves op(A)*X = B.
        /// A is overwritten with its LU factors and B with the solution X, both on the device.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param Ta Operation on matrix A ("N" for no transpose, "T" for transpose).
        /// @param a Input square matrix, replaced with its LU factorization.
        /// @param b Input/output right-hand side matrix, replaced with the solution X.
        template <typename T>
        void SolveGenSysLinEquations(CudaRuntime &cudart, const char *Ta, Matrix_<T> &a, Matrix_<T> &b)
        {
            if (a.shape().first != a.shape().second)
            {
                throw std::runtime_error("A should be square");
            }
            if (b.shape().first != a.shape().first)
            {
                throw std::runtime_error("B should have as much rows as A. B(n, nrhs)");
            }
            cublasOperation_t transa = get_trans(Ta);
            check_device_alloc(cudart, a);
            check_device_alloc(cudart, b);

            int64_t n = static_cast<int64_t>(a.shape().first);
            int64_t nrhs = static_cast<int64_t>(b.shape().second);
            cusolverDnHandle_t handle = cudart.getcuSolverHandle();
            cudaDataType data = std::is_same_v<T, double> ? CUDA_R_64F : CUDA_R_32F;

            cusolverDnParams_t raw_params;
            get_cusolv_error(cusolverDnCreateParams(&raw_params));
            std::unique_ptr<cusolverDnParams, decltype(&cusolverDnDestroyParams)> params(raw_params, &cusolverDnDestroyParams);

            size_t wspacedevice = 0;
            size_t wspacehost = 0;
            get_cusolv_error(cusolverDnXgetrf_bufferSize(handle, params.get(), n, n, data, a.gpu_data(), n, data,
                                                         &wspacedevice, &wspacehost));

            // Device-only buffers: pivot indices and getrf workspace
            Vector<int64_t> ipiv(n, cudart);
            check_device_alloc(cudart, ipiv);
            Vector<T> work_device((wspacedevice + sizeof(T) - 1) / sizeof(T), cudart);
            check_device_alloc(cudart, work_device);
            std::vector<char> work_host(wspacehost);
            Vector<int> info(1, 0);
            info.copy2device(cudart);

            get_cusolv_error(cusolverDnXgetrf(handle, params.get(), n, n, data, a.gpu_data(), n, ipiv.gpu_data(), data,
                                              work_device.gpu_data(), wspacedevice, work_host.data(), wspacehost,
                                              info.gpu_data()));

            info.copy2host(cudart);
            if (info.sum() != 0)
            {
                throw std::runtime_error("Failure in GETRF");
            }
            info.copy2device(cudart);
            get_cusolv_error(cusolverDnXgetrs(handle, params.get(), transa, n, nrhs, data, a.gpu_data(), n, ipiv.gpu_data(),
                                              data, b.gpu_data(), n, info.gpu_data()));
            info.copy2host(cudart);
            if (info.sum() != 0)
            {
                throw std::runtime_error("Failure in GETRS");
            }
        }

        /// @brief Solves a general system of linear equations op(A)*X = B, with the operation on A given last.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param a Input square matrix, replaced with its LU factorization.
        /// @param b Input/output right-hand side matrix, replaced with the solution X.
        /// @param Ta Operation on matrix A ("N" for no transpose, "T" for transpose).
        template <typename T>
        void SolveGenSysLinEquations(CudaRuntime &cudart, Matrix_<T> &a, Matrix_<T> &b, const char *Ta)
        {
            SolveGenSysLinEquations(cudart, Ta, a, b);
        }

        template void SolveGenSysLinEquations<double>(CudaRuntime &cudart, const char *Ta, Matrix_<double> &a, Matrix_<double> &b);
        template void SolveGenSysLinEquations<float>(CudaRuntime &cudart, const char *Ta, Matrix_<float> &a, Matrix_<float> &b);
        template void SolveGenSysLinEquations<double>(CudaRuntime &cudart, Matrix_<double> &a, Matrix_<double> &b, const char *Ta);
        template void SolveGenSysLinEquations<float>(CudaRuntime &cudart, Matrix_<float> &a, Matrix_<float> &b, const char *Ta);

    } // namespace gpu
} // namespace lahva
