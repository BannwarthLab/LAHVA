/// @file additional-level1.cu
/// @brief GPU implementations of additional Level-1 operations (element-wise vector operations).
///
/// Implements specialized GPU kernels for tensor reductions, mixed-precision vector operations,
/// and custom element-wise operations with optimized Kahan summation for Ozaki algorithm.

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
        /// @brief GPU kernel for copying tensor data with optional type conversion.
        template<typename in, typename out>
        __global__ void CopyTensors_(unsigned long size, const in* d_in, out* d_out)
        {
            unsigned long idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size)
                d_out[idx] = d_in[idx];
        }

        /// @brief Host wrapper for type-converting tensor copy.
        template<typename in, typename out>
        void CopyTensors(const unsigned long size, const in* d_in, out* d_out)
        {
            unsigned int blockSize = 512;
            int gridSize = (int)((size + blockSize - 1) / blockSize);
            CopyTensors_<in, out><<<gridSize, blockSize, 0, 0>>>(size, d_in, d_out);
        }

        /// @brief GPU kernel for scaled vector addition (float to double) with FMA.
        ///
        /// Computes: b[i] += alpha * a[i] with float-to-double conversion.
        /// Uses FMA for precision preservation and grid-stride loop for flexibility.
        ///
        /// @param ndim Number of elements.
        /// @param alpha Scaling factor (double precision).
        /// @param a Input vector (single precision).
        /// @param b Output vector (double precision, accumulated).
        __global__ void AddVector_(unsigned long long ndim, const double alpha, const float *a, double *b)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            const size_t stride = blockDim.x * gridDim.x;
            #pragma unroll
            for (size_t i = index; i < ndim; i += stride)
            {
                b[i] = getFMA<double>(a[i], alpha, b[i]);
            }
        };

        /// @brief GPU kernel for Ozaki algorithm summation using Kahan compensated summation.
        ///
        /// Computes accurate sum of scaled vectors: b[i] += sum_j(alphas[j] * as[j][i])
        /// using Kahan compensated summation to minimize rounding errors in iterative refinement.
        ///
        /// @param ndim Number of elements.
        /// @param nsplit Number of input vectors to sum.
        /// @param alphas Scaling factors for each vector.
        /// @param as Array of input vector pointers (single precision).
        /// @param b Output vector (double precision, accumulated with Kahan correction).
        __global__ void BetterSum(unsigned long long ndim, unsigned int nsplit, const double* alphas, const float **as, double *b)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            const size_t stride = blockDim.x * gridDim.x;
            #pragma unroll
            for (size_t i = index; i < ndim; i += stride)
            {
                double sum = 0.0, c = 0.0;
                for (unsigned int j = 0; j < nsplit; j++)
                {
                    double temp = (as[j][i]) * alphas[j];
                    temp -= c;
                    double t = sum + temp;
                    c = (t - sum) - temp;
                    sum = t;
                }
                b[i] += sum;
            }
        };

        /// @brief GPU kernel for scaled vector addition (double to float) with type conversion.
        ///
        /// Computes: b[i] += alpha * a[i] with double-to-float conversion.
        /// Converts high-precision input to single precision before accumulation.
        ///
        /// @param ndim Number of elements.
        /// @param alpha Scaling factor (double precision).
        /// @param a Input vector (double precision).
        /// @param b Output vector (single precision, accumulated).
        __global__ void AddVector_(unsigned long long ndim, const double alpha, const double *a, float *b)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            const size_t stride = blockDim.x * gridDim.x;
            #pragma unroll
            for (size_t i = index; i < ndim; i += stride)
            {
                b[i] += static_cast<float>(a[i] * alpha);
            }
        };

        /// @brief GPU kernel for scaled vector addition (half to double) with type conversion.
        ///
        /// Computes: b[i] += alpha * a[i] with half-to-double conversion.
        /// Converts low-precision input to double precision before accumulation.
        ///
        /// @param ndim Number of elements.
        /// @param alpha Scaling factor (double precision).
        /// @param a Input vector (half precision).
        /// @param b Output vector (double precision, accumulated).
        __global__ void AddVector_(unsigned long long ndim, const double alpha, const __half *a, double *b)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            const size_t stride = blockDim.x * gridDim.x;
            #pragma unroll
            for (size_t i = index; i < ndim; i += stride)
            {
                b[i] += static_cast<double>(a[i]) * alpha;
            }
        };


        /// @brief Host wrapper for Ozaki algorithm summation with Kahan compensation.
        ///
        /// Launches GPU kernel for accurate summation of multiple scaled vectors
        /// using compensated summation for iterative refinement.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param ndim Number of elements.
        /// @param nsplit Number of input vectors.
        /// @param alphas Scaling factors (on device).
        /// @param as Array of input vector pointers (on device).
        /// @param b Output vector (on device, accumulated).
        void MergeOzaki(const CudaRuntime& cudart, unsigned long long ndim, unsigned int nsplit, const double* alphas, const float** as, double* b)
        {
            unsigned long long blockSize = cudart.blockSize();
            BetterSum<<<cudart.gridSize(ndim, 1), blockSize, 0, cudart.getStream()>>>(ndim, nsplit, alphas, as, b);
        }

        /// @brief Host wrapper for scaled vector addition (float to double).
        ///
        /// Launches GPU kernel with grid-stride loop for mixed-precision accumulation.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param ndim Number of elements.
        /// @param alpha Scaling factor.
        /// @param a Input tensor (single precision).
        /// @param b Output tensor (double precision, accumulated).
        void AddVector(const CudaRuntime& cudart, unsigned long long ndim, const double alpha, const Tensor_<float>& a, Tensor_<double>& b)
        {
            check_device_alloc(cudart, a);
            check_device_alloc(cudart, b);
            unsigned long long blockSize = cudart.blockSize();
            AddVector_<<<std::ceil(static_cast<double>(cudart.gridSize(ndim, 1)) / 4), blockSize, 0, cudart.getStream()>>>(ndim, alpha, a.gpu_data(), b.gpu_data());
        }

        /// @brief Host wrapper for scaled vector addition (double to float).
        ///
        /// Launches GPU kernel for mixed-precision accumulation with conversion.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param ndim Number of elements.
        /// @param alpha Scaling factor.
        /// @param a Input tensor (double precision).
        /// @param b Output tensor (single precision, accumulated).
        void AddVector(const CudaRuntime& cudart, unsigned long long ndim, const double alpha, const Tensor_<double>& a, Tensor_<float>& b)
        {
            check_device_alloc(cudart, a);
            check_device_alloc(cudart, b);
            unsigned long long blockSize = cudart.blockSize();
            AddVector_<<<cudart.gridSize(ndim, 1), blockSize, 0, cudart.getStream()>>>(ndim, alpha, a.gpu_data(), b.gpu_data());
        }

        /// @brief Host wrapper for scaled vector addition (half to double).
        ///
        /// Launches GPU kernel for mixed-precision accumulation from half-precision input.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param ndim Number of elements.
        /// @param alpha Scaling factor.
        /// @param a Input tensor (half precision).
        /// @param b Output tensor (double precision, accumulated).
        void AddVector(const CudaRuntime& cudart, unsigned long long ndim, const double alpha, const Tensor_<__half>& a, Tensor_<double>& b)
        {
            check_device_alloc(cudart, a);
            check_device_alloc(cudart, b);
            unsigned long long blockSize = cudart.blockSize();
            AddVector_<<<cudart.gridSize(ndim, 1), blockSize, 0, cudart.getStream()>>>(ndim, alpha, a.gpu_data(), b.gpu_data());
        }

        /// @brief Host wrapper to apply unary operation element-wise to tensor.
        ///
        /// Launches GPU kernel to apply custom operation (e.g., absolute value) to each element.
        ///
        /// @tparam T Element type.
        /// @tparam op Unary operation functor type.
        /// @param cudart CUDA runtime instance.
        /// @param in Input/output tensor (modified in-place).
        /// @param operation Unary operation functor.
        template<typename T, class op>
        void ApplyKernel(const CudaRuntime& cudart, Tensor_<T>& in, op operation)
        {
            check_device_alloc(cudart, in);
            ApplyKernel_<<<cudart.gridSize(in.size(), 1), cudart.blockSize(), 0, cudart.getStream()>>>(in.size(), in.gpu_data(), operation);
        }

        /// @brief Host wrapper to compute sum of all tensor elements.
        ///
        /// Uses parallel reduction with add_rn functor for correctly-rounded addition.
        /// Returns result copied back to host.
        ///
        /// @tparam T Element type.
        /// @param cudart CUDA runtime instance.
        /// @param in Input tensor.
        /// @param res Temporary tensor for reduction (updated in-place).
        /// @return Sum of all elements.
        template<typename T>
        T Sum_(const CudaRuntime& cudart, const Tensor_<T>& in, Tensor_<T>& res)
        {
            check_device_alloc(cudart, in);
            check_device_alloc(cudart, res);
            unsigned long long blockSize = cudart.blockSize();
            GPUReduction<T, add_rn<T>>(cudart, in.size(), in.gpu_data(), res.gpu_data(), blockSize, add_rn<T>());
            res.copy2host(cudart);
            cudart.synchronize();
            return res[0];
        }

        /// @brief Host wrapper to find maximum element in tensor.
        ///
        /// Uses parallel reduction with max_ functor.
        /// Returns result copied back to host.
        ///
        /// @tparam T Element type.
        /// @param cudart CUDA runtime instance.
        /// @param in Input tensor.
        /// @param res Temporary tensor for reduction (updated in-place).
        /// @return Maximum element value.
        template<typename T>
        T MaxElement_(const CudaRuntime& cudart, const Tensor_<T>& in, Tensor_<T>& res)
        {
            check_device_alloc(cudart, in);
            check_device_alloc(cudart, res);
            unsigned long long blockSize = cudart.blockSize();
            GPUReduction<T, max_<T>>(cudart, in.size(), in.gpu_data(), res.gpu_data(), blockSize);
            res.copy2host(cudart);
            cudart.synchronize();
            return res[0];
        }

        /// @brief Host wrapper to find minimum element in tensor.
        ///
        /// Uses parallel reduction with min_ functor.
        /// Returns result copied back to host.
        ///
        /// @tparam T Element type.
        /// @param cudart CUDA runtime instance.
        /// @param in Input tensor.
        /// @param res Temporary tensor for reduction (updated in-place).
        /// @return Minimum element value.
        template<typename T>
        T MinElement_(const CudaRuntime& cudart, const Tensor_<T>& in, Tensor_<T>& res)
        {
            check_device_alloc(cudart, in);
            check_device_alloc(cudart, res);
            unsigned long long blockSize = cudart.blockSize();
            GPUReduction<T, min_<T>>(cudart, in.size(), in.gpu_data(), res.gpu_data(), blockSize);
            res.copy2host(cudart);
            cudart.synchronize();
            return res[0];
        }


        template float MaxElement_<float>(const CudaRuntime& cudart, const Tensor_<float>& in, Tensor_<float>& res);
        template double MaxElement_<double>(const CudaRuntime& cudart, const Tensor_<double>& in, Tensor_<double>& res);
        template float MinElement_<float>(const CudaRuntime& cudart, const Tensor_<float>& in, Tensor_<float>& res);
        template double MinElement_<double>(const CudaRuntime& cudart, const Tensor_<double>& in, Tensor_<double>& res);
        template float Sum_<float>(const CudaRuntime& cudart, const Tensor_<float>& in, Tensor_<float>& res);
        template double Sum_<double>(const CudaRuntime& cudart, const Tensor_<double>& in, Tensor_<double>& res);
        template void ApplyKernel<float, fabs_gpu<float>>(const CudaRuntime& cudart, Tensor_<float>& in, fabs_gpu<float> operation);
        template void ApplyKernel<double, fabs_gpu<double>>(const CudaRuntime& cudart, Tensor_<double>& in, fabs_gpu<double> operation);
        template void CopyTensors<double, double>(const unsigned long size, const double* d_in, double* d_out);
        template void CopyTensors<float, float>(const unsigned long size, const float* d_in, float* d_out);
        template void CopyTensors<double, float>(const unsigned long size, const double* d_in, float* d_out);
        template void CopyTensors<float, double>(const unsigned long size, const float* d_in, double* d_out);
        template void CopyTensors<int, int>(const unsigned long size, const int* d_in, int* d_out);
        template void CopyTensors<__half, __half>(const unsigned long size, const __half* d_in, __half* d_out);
    } // namespace gpu
    
} // namespace lahva
