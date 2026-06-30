/// @file norm.cuh
/// @brief GPU kernels for computing matrix norms (Frobenius and Frobenius difference).
///
/// Provides CUDA kernels for efficient parallel computation of Frobenius norms
/// using parallel reduction with per-block partial sums.

#pragma once
#include "../../gpu-utils/utils.hpp"
#include "../reductions/reduction.cuh"
#include "common.h"

namespace lahva
{
    namespace gpu
    {
        /// @brief GPU kernel to compute Frobenius norm squared using parallel reduction.
        ///
        /// Computes the sum of squared matrix elements: sum(mat[i]^2).
        /// Uses Fused-Multiply-Add (FMA) for precision and shared memory reduction with per-block partial sums.
        /// Block size must be a power of 2 for the reduction algorithm.
        ///
        /// @tparam blockSize CUDA block size (template parameter for optimization).
        /// @tparam T Floating-point type (float or double).
        /// @param size Total number of elements in matrix.
        /// @param mat Input matrix.
        /// @param sum Output vector of partial sums (one per block).
        template <size_t blockSize, typename T>
        __global__ static void Frobenius_(const unsigned long long size, const T *mat, T *sum)
        {
            __shared__ T tmp_[blockSize];
            unsigned long long tid = threadIdx.x;
            unsigned long long id = blockIdx.x * blockDim.x + tid;

            T tmp = 0.0;
            if (id < size)
            {
                tmp = getFMA<T>(mat[id], mat[id], (T)0.0);
            }
            tmp_[tid] = tmp;
            __syncthreads();

            #pragma unroll
            for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
            {
                if (tid < stride)
                {
                    tmp_[tid] = getAdd<T>(tmp_[tid + stride], tmp_[tid]);
                }
                __syncthreads();
            }

            if (tid == 0)
                sum[blockIdx.x] = tmp_[0];
        }

        /// @brief GPU kernel to compute squared Frobenius norm of difference using parallel reduction.
        ///
        /// Computes the sum of squared element-wise differences: sum((A[i] - B[i])^2).
        /// Uses Fused-Multiply-Add (FMA) for precision and shared memory reduction with per-block partial sums.
        /// Block size must be a power of 2 for the reduction algorithm.
        ///
        /// @tparam blockSize CUDA block size (template parameter for optimization).
        /// @tparam T Floating-point type (float or double).
        /// @param size Total number of elements in matrices.
        /// @param A First input matrix.
        /// @param B Second input matrix.
        /// @param result Output vector of partial sums (one per block).
        template <size_t blockSize, typename T>
        __global__ void Frobenius_2(const unsigned long long size, const T* A, const T* B, T* result)
        {
            __shared__ T sharedData[blockSize];
            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            T sum = 0.0;
            if (idx < size)
            {
                T diff = getSub<T>(A[idx], B[idx]);
                sum = getFMA<T>(diff, diff, 0.0);
            }

            sharedData[threadIdx.x] = sum;
            __syncthreads();

            #pragma unroll
            for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
            {
                if (threadIdx.x < stride)
                {
                    sharedData[threadIdx.x] = getAdd<T>(sharedData[threadIdx.x + stride], sharedData[threadIdx.x]);
                }
                __syncthreads();
            }

            if (threadIdx.x == 0)
            {
                result[blockIdx.x] = sharedData[0];
            }
        }

    } // namespace gpu
} //namespace lahva