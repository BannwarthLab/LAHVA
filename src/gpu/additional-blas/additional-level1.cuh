/// @file additional-level1.cuh
/// @brief GPU kernel for applying unary operations to tensors.
///
/// Provides a generic GPU kernel template for applying custom unary operations
/// element-wise to tensors.

#pragma once
#include "runtime.hpp"
namespace lahva
{
    namespace gpu
    {
        template<typename T, class op>
        __global__ void ApplyKernel_(unsigned long long ndim, T *a, op func)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < ndim)
            {
                a[index] = func(a[index]);
            }
        
        };

        // template<typename in, typename out>
        // __global__ void CopyTensors_(const unsigned long size, const in* d_in, out* d_out)
        // {
        //     int index = blockIdx.x * blockDim.x + threadIdx.x;
        //     if (index < size)
        //     {
        //         d_out[index] = d_in[index];
        //     } 
        // }
        

    } // namespace gpu
    
} // namespace lahva