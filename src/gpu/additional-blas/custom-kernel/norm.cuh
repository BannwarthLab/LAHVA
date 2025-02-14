#pragma once
#include "../../../gpu-utils/utils.hpp"
#include "common.h"
namespace lahva
{
    namespace gpu
    {

        template <size_t blockSize, typename T>
        __global__ static void Frobenius_(const unsigned long long size, const T *mat, T *sum)
        {
            __shared__ T tmp_[blockSize];
            unsigned long long tid = threadIdx.x;
            unsigned long long id = blockIdx.x * blockDim.x + tid;
            //size_t gridSize = blockSize * gridDim.x;

            
    
            //extern __shared__ T tmp_[blocks];
            // due to the shared directive every block gets its own copy of tmp
            T tmp = 0.0;
            if (id < size)
            {
                tmp = getFMA<T>(mat[id], mat[id], 0.0);
                //tmp += mat[id] * mat[id];
            }
            tmp_[tid] = 0;
            tmp_[tid] = tmp;
            __syncthreads();

            // for reductions, threadsPerBlock must be a power of 2
            // because of the following code
            for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
            {
                if (tid < stride)
                {
                    tmp_[tid] += tmp_[tid + stride];
                }
                __syncthreads();
            }
            
            // Thread 0 adds partial sums to overall sum
            if (tid == 0)
                sum[blockIdx.x] = tmp_[0];

            //reduceCUDA<blockSize, T, add_rn<T>>(sum, sum, blockSize, add_rn<T>());
        }

        template <size_t blockSize, typename T>
        __global__ void Frobenius_2(const unsigned long long size, const T* A, const T* B, T* result) {
        __shared__ T sharedData[blockSize]; // Adjust size based on threads per block
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        T sum = 0.0f;
        if (idx < size) {
            T diff = getSub<T>(A[idx], B[idx]);
            sum = getFMA<T>(diff, diff, 0.0); // Compute squared difference
        }

        // Store the partial sum in shared memory
        sharedData[threadIdx.x] = sum;
        __syncthreads(); // Ensure all threads have written to shared memory

        // Perform reduction in shared memory
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                sharedData[threadIdx.x] = getAdd<T>(sharedData[threadIdx.x + stride], sharedData[threadIdx.x]);
            }
            __syncthreads(); // Ensure all threads have completed the addition
        }

        // Write the block's partial sum to global memory
        if (threadIdx.x == 0) {
            result[blockIdx.x] = sharedData[0];
        }
}

    } // namespace gpu
    
}