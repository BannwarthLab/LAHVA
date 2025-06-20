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