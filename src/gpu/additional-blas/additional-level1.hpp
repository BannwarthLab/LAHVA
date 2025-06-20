#pragma once
#include <cuda_runtime.h>
namespace lahva
{
    namespace gpu
    {

        template<typename in, typename out>
        __global__ void CopyTensors_(unsigned long size, const in* d_in, out* d_out);

        template<typename in, typename out>
        void CopyTensors(const unsigned long size, const in* d_in, out* d_out);

    } // namespace gpu
} // namespace lahva
