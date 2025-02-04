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
        

    } // namespace gpu
    
} // namespace lahva
