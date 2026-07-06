/// @file reduction.cuh
/// @brief GPU parallel reduction kernels with custom operations.
///
/// Provides highly-optimized parallel reduction algorithms using shared memory,
/// warp-level operations, and custom reduction functors for efficient GPU aggregation.

#pragma once
#include "impl/gpu/utils.hpp"

namespace lahva
{
    namespace gpu
    {
        /// @brief Device-level warp reduction for final synchronization step.
        ///
        /// Reduces values within a warp (32 threads) using the warp shuffle approach.
        /// Assumes block size is power of 2. Only threads < 32 should call this.
        ///
        /// @tparam blockSize CUDA block size (compile-time for unrolling).
        /// @tparam T Data type for reduction.
        /// @tparam op Binary reduction operation functor.
        /// @param sdata Shared memory array containing values to reduce.
        /// @param tid Thread ID within block.
        /// @param func Reduction operation functor.
        template <size_t blockSize, typename T, class op>
        __device__ void warpReduce(volatile T *sdata, size_t tid, op func)
        {
            if (blockSize >= 64)
                sdata[tid] = func(sdata[tid], sdata[tid + 32]);
            if (blockSize >= 32)
                sdata[tid] = func(sdata[tid], sdata[tid + 16]);
            if (blockSize >= 16)
                sdata[tid] = func(sdata[tid], sdata[tid + 8]);
            if (blockSize >= 8)
                sdata[tid] = func(sdata[tid], sdata[tid + 4]);
            if (blockSize >= 4)
                sdata[tid] = func(sdata[tid], sdata[tid + 2]);
            if (blockSize >= 2)
                sdata[tid] = func(sdata[tid], sdata[tid + 1]);
        }

        /// @brief GPU kernel for parallel reduction of array elements.
        ///
        /// Reduces input array using parallel reduction within blocks and across grid.
        /// Produces per-block partial sums that can be further reduced.
        /// Block size must be power of 2 and template parameter.
        ///
        /// @tparam blockSize CUDA block size (template parameter for optimization).
        /// @tparam T Data type for reduction.
        /// @tparam op Binary reduction operation functor.
        /// @param g_idata Input array.
        /// @param g_odata Output array of per-block partial sums.
        /// @param n Number of elements to reduce.
        /// @param func Reduction operation functor.
        template <size_t blockSize, typename T, class op>
        __global__ void reduceCUDA(const T *g_idata, T *g_odata, size_t n, op func)
    {
        __shared__ T sdata[blockSize];

        size_t tid = threadIdx.x;
        size_t i = blockIdx.x * blockSize + tid;
        size_t gridSize = blockSize * gridDim.x;
        sdata[tid] = func.ini_value();

        while (i < n)
        {
            sdata[tid] = func(sdata[tid], g_idata[i]);
            i += gridSize;
        }
        __syncthreads();

        if (blockSize >= 1024)
        {
            if (tid < 512)
            {
                sdata[tid] = func(sdata[tid], sdata[tid+512]);
            }
            __syncthreads();
        }
        if (blockSize >= 512)
        {
            if (tid < 256)
            {
                sdata[tid] = func(sdata[tid], sdata[tid+256]);
            }
            __syncthreads();
        }
        if (blockSize >= 256)
        {
            if (tid < 128)
            {
                sdata[tid] = func(sdata[tid], sdata[tid+128]);
            }
            __syncthreads();
        }
        if (blockSize >= 128)
        {
            if (tid < 64)
            {
                sdata[tid] = func(sdata[tid], sdata[tid+64]);
            }
            __syncthreads();
        }

        if (tid < 32)
            warpReduce<blockSize, T, op>(sdata, tid, func);
        if (tid == 0)
            g_odata[blockIdx.x] = sdata[0];
    }

    /// @brief Alternative GPU kernel for parallel reduction (unused/experimental).
    ///
    /// @tparam blockSize CUDA block size.
    /// @tparam T Data type for reduction.
    /// @tparam op Binary reduction operation functor.
    /// @param g_idata Input array.
    /// @param g_odata Output array.
    /// @param n Number of elements.
    /// @param func Reduction operation.
    template <size_t blockSize, typename T, class op>
    __global__ void reduceCUDA_(const T *g_idata, T *g_odata, size_t n, op func)
    {
        size_t tid = threadIdx.x;
        size_t i = blockIdx.x * blockSize + tid;
        size_t gridSize = blockSize * gridDim.x;
        __shared__ T sdata[blockSize];
        sdata[tid] = func.ini_value();

        if (i < n)
        {
            sdata[tid] = g_idata[i];
            __syncthreads();

            for (int stride = 1; stride < blockSize; stride *= 2)
            {
                if (tid < stride)
                {
                    sdata[tid] = func(sdata[tid], sdata[tid + stride]);
                }
                __syncthreads();
            }
        }
        if (tid == 0)
            g_odata[blockIdx.x] = sdata[0];
    }

    /// @brief Host-level reduction driver with multi-level GPU reduction.
    ///
    /// Performs multi-pass reduction: repeatedly calls reduceCUDA kernel
    /// until result fits in a single block, then does final reduction.
    /// Block size is a template parameter for optimal compilation.
    ///
    /// @tparam blockSize CUDA block size (template parameter for optimization).
    /// @tparam T Data type for reduction.
    /// @tparam op Binary reduction operation functor.
    /// @param cudart CUDA runtime instance.
    /// @param ndim Number of input elements.
    /// @param dA Input array on device.
    /// @param dRes Output array on device (also used as scratch space).
    /// @param func Reduction operation functor.
    template <size_t blockSize, typename T, class op>
    void GPUReduction_(const CudaRuntime &cudart, const unsigned long long ndim, const T* dA, T* dRes, op func)
    {
        T tot = 0.;
        size_t n = ndim;
        size_t blocksPerGrid = std::ceil((1. * n) / blockSize);

        T *tmp = dRes;

        const T *from = dA;

        do
        {
            blocksPerGrid = std::ceil((1. * n) / blockSize);
            reduceCUDA<blockSize><<<blocksPerGrid, blockSize, 0, cudart.getStream()>>>(from, tmp, n, func);
            from = tmp;
            n = blocksPerGrid;
        } while (n > blockSize);
        if (n > 1)
            reduceCUDA<blockSize><<<1, blockSize, 0, cudart.getStream()>>>(tmp, tmp, n, func);

    }

    /// @brief Host dispatcher for GPU reduction with runtime block size selection.
    ///
    /// Selects and launches appropriate block-size-specialized GPU reduction kernel.
    /// Supports various block sizes from 1 to 1024 threads.
    ///
    /// @tparam T Data type for reduction.
    /// @tparam op Binary reduction operation functor.
    /// @param cudart CUDA runtime instance.
    /// @param ndim Number of input elements.
    /// @param dA Input array on device.
    /// @param dRes Output array on device (also used as scratch space).
    /// @param blockSize CUDA block size for kernel launch (must match a supported case).
    /// @param func Reduction operation functor (default-constructed if not provided).
    template<typename T, class op>
    void GPUReduction(const CudaRuntime& cudart, const unsigned long long ndim, const T* dA, T* dRes, size_t blockSize, op func = op())
    {
        switch (blockSize)
        {
        case 1024:
            GPUReduction_<1024, T>(cudart, ndim, dA, dRes, func);
            break;
        case 512:
            GPUReduction_<512, T>(cudart, ndim, dA, dRes, func);
            break;
        case 480:
            GPUReduction_<480, T>(cudart, ndim, dA, dRes, func);
            break;
        case 448:
            GPUReduction_<448, T>(cudart, ndim, dA, dRes, func);
            break;
        case 416:
            GPUReduction_<416, T>(cudart, ndim, dA, dRes, func);
            break;
        case 384:
            GPUReduction_<384, T>(cudart, ndim, dA, dRes, func);
            break;
        case 352:
            GPUReduction_<352, T>(cudart, ndim, dA, dRes, func);
            break;
        case 320:
            GPUReduction_<320, T>(cudart, ndim, dA, dRes, func);
            break;
        case 288:
            GPUReduction_<288, T>(cudart, ndim, dA, dRes, func);
            break;
        case 256:    
            GPUReduction_<256, T>(cudart, ndim, dA, dRes, func);
            break;
        case 224:
            GPUReduction_<224, T>(cudart, ndim, dA, dRes, func);
            break;
        case 192:
            GPUReduction_<192, T>(cudart, ndim, dA, dRes, func);
            break;
        case 160:
            GPUReduction_<160, T>(cudart, ndim, dA, dRes, func);
            break;
        case 128:
            GPUReduction_<128, T>(cudart, ndim, dA, dRes, func);
            break;
        case 64:
            GPUReduction_<64, T>(cudart, ndim, dA, dRes, func);
            break;
        case 32:
            GPUReduction_<32, T>(cudart, ndim, dA, dRes, func);
            break;
        case 16:
            GPUReduction_<16, T>(cudart, ndim, dA, dRes, func);
            break;
        case 8:
            GPUReduction_<8, T>(cudart, ndim, dA, dRes, func);
            break;
        case 4:
            GPUReduction_<4, T>(cudart, ndim, dA, dRes, func);
            break;
        case 2:
            GPUReduction_<2, T>(cudart, ndim, dA, dRes, func);
            break;
        case 1:
            GPUReduction_<1, T>(cudart, ndim, dA, dRes, func);
            break;
        default:
            break;
        }
        
    }
    } // namespace gpu
} // namespace lahva
