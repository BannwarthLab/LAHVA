#pragma once
#include "runtime.hpp"
namespace lahva
{
    namespace gpu
    {
        template<typename T>
        class fabs_gpu
        {
        public:
            __device__ T operator()(T x)
            {
                return fabs(x);
            }
        };

        template<typename T, class op>
        __global__ void ApplyKernel_(unsigned long long ndim, T *a, op func)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < ndim)
            {
                a[index] = func(a[index]);
            }
        
        };

        

    } // namespace gpu
    
} // namespace lahva
