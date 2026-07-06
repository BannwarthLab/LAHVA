/// @file additional-level2.cu
/// @brief GPU implementations of additional Level-2 operations (matrix operations).
///
/// Implements specialized GPU kernels for matrix addition with mixed-precision support
/// and optimized 1D/2D blocking strategies for various type combinations.

#include "impl/blas/gpu/additional-level1.hpp"
#include "additional-level1.cuh"
#include "custom-kernel/common.h"
#include "impl/blas/gpu/additional-level1.hpp"
#include "linalg.hpp"
#include "reductions/common.cuh"
#include "reductions/reduction.cuh"
#include "runtime.hpp"
namespace lahva
{
    namespace gpu
    {
        /// @brief GPU kernel for 2D matrix addition (float to double) with FMA.
        ///
        /// Element-wise scaled addition for 2D matrices: b[row,col] += alpha * a[row,col]
        /// with float-to-double conversion and FMA for precision.
        /// Uses 2D grid with different dimensions for input and output matrices.
        ///
        /// @param nrows_a Number of rows in input matrix a.
        /// @param ncols_a Number of columns in input matrix a.
        /// @param nrows_b Number of rows in output matrix b.
        /// @param ncols_b Number of columns in output matrix b.
        /// @param alpha Scaling factor (double precision).
        /// @param a Input matrix (single precision, row-major).
        /// @param b Output matrix (double precision, row-major, accumulated).
        __global__ void AddVector2D_(unsigned long long nrows_a, unsigned long long ncols_a, unsigned long long nrows_b, unsigned long long ncols_b, const double alpha, const float *a, double *b)
        {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            if (row < nrows_b && col < ncols_b)
            {
                b[row * ncols_b + col] = getFMA<double>(a[row * ncols_a + col], alpha, b[row * ncols_b + col]);
            }
        };

        /// @brief GPU kernel for 2D matrix addition (float to float) with scalar FMA.
        ///
        /// Element-wise scaled addition for 2D single-precision matrices with FMA.
        /// Uses 2D grid for efficient thread-to-element mapping.
        ///
        /// @param nrows_a Number of rows in input matrix.
        /// @param ncols_a Number of columns in input matrix.
        /// @param nrows_b Number of rows in output matrix.
        /// @param ncols_b Number of columns in output matrix.
        /// @param alpha Scaling factor (single precision).
        /// @param a Input matrix (single precision, row-major).
        /// @param b Output matrix (single precision, row-major, accumulated).
        __global__ void AddVector2D_(unsigned long long nrows_a, unsigned long long ncols_a, unsigned long long nrows_b, unsigned long long ncols_b, const float alpha, const float *a, float *b)
        {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            if (row < nrows_b && col < ncols_b)
            {
                b[row * ncols_b + col] = getFMA<float>(a[row * ncols_a + col], alpha, b[row * ncols_b + col]);
            }
        };

        /// @brief GPU kernel for 2D matrix addition with device-allocated scalar (float to double).
        ///
        /// Element-wise scaled addition where scalar alpha is on device memory.
        /// Avoids copying scalar to constant memory.
        ///
        /// @param nrows_a Number of rows in input matrix.
        /// @param ncols_a Number of columns in input matrix.
        /// @param nrows_b Number of rows in output matrix.
        /// @param ncols_b Number of columns in output matrix.
        /// @param alpha Pointer to scaling factor on device (double precision).
        /// @param a Input matrix (single precision, row-major).
        /// @param b Output matrix (double precision, row-major, accumulated).
        __global__ void AddVector2D_(unsigned long long nrows_a, unsigned long long ncols_a, unsigned long long nrows_b, unsigned long long ncols_b, const double * alpha, const float *a, double *b)
        {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            if (row < nrows_b && col < ncols_b)
            {
                b[row * ncols_b + col] = getFMA<double>(a[row * ncols_a + col], *alpha, b[row * ncols_b + col]);
            }
        };

        /// @brief GPU kernel for optimized 1D matrix addition using vector operations (float to double).
        ///
        /// Specialized kernel for square matrices using float2/double2 vector instructions.
        /// Processes two elements per thread for improved bandwidth utilization.
        /// Only used when input and output matrices have same shape.
        ///
        /// @param ntot Total number of elements (must be even).
        /// @param alpha Pointer to scaling factor on device (double precision).
        /// @param a Input matrix (single precision, row-major).
        /// @param b Output matrix (double precision, row-major, accumulated).
        __global__ void AddVector1D_(unsigned long long ntot, const double * alpha, const float *a, double *b)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            const size_t stride = blockDim.x * gridDim.x;
            for (size_t i = idx; i < ntot / 2; i += stride)
            {
                float2 val_a = reinterpret_cast<const float2*>(a)[i];
                double2 val_a_d = reinterpret_cast<double2*>(b)[i];
                val_a_d.x += static_cast<double>(val_a.x) * (*alpha);
                val_a_d.y += static_cast<double>(val_a.y) * (*alpha);
                reinterpret_cast<double2*>(b)[i] = val_a_d;
            }
        };

        /// @brief GPU kernel for 2D matrix addition with device-allocated scalar (float to float).
        ///
        /// Element-wise scaled addition for single-precision matrices with scalar in device memory.
        ///
        /// @param nrows_a Number of rows in input matrix.
        /// @param ncols_a Number of columns in input matrix.
        /// @param nrows_b Number of rows in output matrix.
        /// @param ncols_b Number of columns in output matrix.
        /// @param alpha Pointer to scaling factor on device (single precision).
        /// @param a Input matrix (single precision, row-major).
        /// @param b Output matrix (single precision, row-major, accumulated).
        __global__ void AddVector2D_(unsigned long long nrows_a, unsigned long long ncols_a, unsigned long long nrows_b, unsigned long long ncols_b, const float * alpha, const float *a, float *b)
        {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            if (row < nrows_b && col < ncols_b)
            {
                b[row * ncols_b + col] = getFMA<float>(a[row * ncols_a + col], *alpha, b[row * ncols_b + col]);
            }
        };

        /// @brief Host wrapper for matrix addition with scalar (float to double).
        ///
        /// Launches 2D grid kernel with 16x16 blocks for mixed-precision matrix addition.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param alpha Scaling factor (double precision, scalar on host).
        /// @param a Input matrix (single precision).
        /// @param b Output matrix (double precision, accumulated).
        void AddMatrix(const CudaRuntime& cudart, const double alpha, const Matrix_<float>& a, Matrix_<double>& b)
        {
            check_device_alloc(cudart, a);
            check_device_alloc(cudart, b);
            cudart.setDevice();
            dim3 blockSize(16, 16);
            dim3 gridSize(ceil((double)(a.shape().second) / double(blockSize.x)), ceil((double)(a.shape().first) / double(blockSize.y)));
            AddVector2D_<<<gridSize, blockSize, 0, cudart.getStream()>>>(a.shape().first, a.shape().second, b.shape().first, b.shape().second, alpha, a.gpu_data(), b.gpu_data());
        }

        /// @brief Host wrapper for matrix addition with scalar (float to float).
        ///
        /// Launches 2D grid kernel with 16x16 blocks for single-precision matrix addition.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param alpha Scaling factor (single precision, scalar on host).
        /// @param a Input matrix (single precision).
        /// @param b Output matrix (single precision, accumulated).
        void AddMatrix(const CudaRuntime& cudart, const float alpha, const Matrix_<float>& a, Matrix_<float>& b)
        {
            check_device_alloc(cudart, a);
            check_device_alloc(cudart, b);
            cudart.setDevice();
            dim3 blockSize(16, 16);
            dim3 gridSize(ceil((double)(a.shape().second) / double(blockSize.x)), ceil((double)(a.shape().first) / double(blockSize.y)));
            AddVector2D_<<<gridSize, blockSize, 0, cudart.getStream()>>>(a.shape().first, a.shape().second, b.shape().first, b.shape().second, alpha, a.gpu_data(), b.gpu_data());
        }

        /// @brief Host wrapper for matrix addition with device-allocated scalar (float to double).
        ///
        /// Selects optimized 1D kernel for square matrices or 2D kernel for general matrices.
        /// 1D kernel uses vector instructions for better bandwidth when matrices have same shape.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param alpha Pointer to scaling factor on device (double precision).
        /// @param a Input matrix (single precision).
        /// @param b Output matrix (double precision, accumulated).
        void AddMatrix(const CudaRuntime& cudart, const double* alpha, const Matrix_<float>& a, Matrix_<double>& b)
        {
            check_device_alloc(cudart, a);
            check_device_alloc(cudart, b);
            cudart.setDevice();
            if (a.shape() == b.shape())
            {
                size_t blockSize = cudart.blockSize();
                size_t numBlocks = cudart.gridSize(a.size(), 1);
                AddVector1D_<<<numBlocks / 4, blockSize, 0, cudart.getStream()>>>(a.size(), alpha, a.gpu_data(), b.gpu_data());
                return;
            }
            dim3 blockSize(16, 32);
            dim3 gridSize(ceil((double)(a.shape().second) / double(blockSize.x)), ceil((double)(a.shape().first) / double(blockSize.y)));
            AddVector2D_<<<gridSize, blockSize, 0, cudart.getStream()>>>(a.shape().first, a.shape().second, b.shape().first, b.shape().second, alpha, a.gpu_data(), b.gpu_data());
        }

        /// @brief Host wrapper for matrix addition with device-allocated scalar (float to float).
        ///
        /// Launches 2D grid kernel with 16x32 blocks for single-precision matrix addition.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param alpha Pointer to scaling factor on device (single precision).
        /// @param a Input matrix (single precision).
        /// @param b Output matrix (single precision, accumulated).
        void AddMatrix(const CudaRuntime& cudart, const float* alpha, const Matrix_<float>& a, Matrix_<float>& b)
        {
            check_device_alloc(cudart, a);
            check_device_alloc(cudart, b);
            cudart.setDevice();
            dim3 blockSize(16, 32);
            dim3 gridSize(ceil((double)(a.shape().second) / double(blockSize.x)), ceil((double)(a.shape().first) / double(blockSize.y)));
            AddVector2D_<<<gridSize, blockSize, 0, cudart.getStream()>>>(a.shape().first, a.shape().second, b.shape().first, b.shape().second, alpha, a.gpu_data(), b.gpu_data());
        }
    } // namespace gpu
    
} // namespace lahva
