#pragma once
#include "../../../gpu-utils/utils.hpp"

namespace lahva
{
    namespace gpu
    {
    template <size_t blockSize, typename T, class op>
    __device__ void warpReduce(volatile T *sdata, size_t tid, op func)
    {
        if (blockSize >= 64)
            sdata[tid] = func(sdata[tid], sdata[tid+32]);
        if (blockSize >= 32)
            sdata[tid] = func(sdata[tid], sdata[tid+16]);
        if (blockSize >= 16)
            sdata[tid] = func(sdata[tid], sdata[tid+8]);
        if (blockSize >= 8)
            sdata[tid] = func(sdata[tid], sdata[tid+4]);
        if (blockSize >= 4)
            sdata[tid] = func(sdata[tid], sdata[tid+2]);
        if (blockSize >= 2)
            sdata[tid] = func(sdata[tid], sdata[tid+1]);
    }

    template <size_t blockSize, typename T, class op>
    __global__ void reduceCUDA(const T *g_idata, T *g_odata, size_t n, op func)
    {
        __shared__ T sdata[blockSize];

        size_t tid = threadIdx.x;
        size_t i = blockIdx.x * (blockSize) + tid;
        size_t gridSize = blockSize * gridDim.x;
        sdata[tid] = 0;

        while (i < n)
        {
            sdata[tid] = func(sdata[tid], g_idata[i]);
            i += gridSize;
        }
        // while (i < n) { sdata[tid] += g_idata[i] + g_idata[i+blockSize]; i += gridSize; }
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
                sdata[tid] = func(sdata[tid], sdata[tid+216]);
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

    template<typename T, class op>
    void GPUReduction(const CudaRuntime& cudart, const unsigned long long ndim, const T* dA, T* dRes, size_t blockSize, op func)
    {   
        switch (blockSize)
        {
        case 1024:
            GPUReduction_<1024, T>(cudart, ndim, dA, dRes, func);
            break;
        case 512:
            GPUReduction_<512, T>(cudart, ndim, dA, dRes, func);
            break;
        case 256:    
            GPUReduction_<256, T>(cudart, ndim, dA, dRes, func);
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
    }
} // namespace name
