#ifndef TCGMTENSOR_ADD_LEVEL1_HPP
#define TCGMTENSOR_ADD_LEVEL1_HPP
#include "linalg.hpp"
#include "runtime.hpp"

namespace tcgmtensor{
    namespace gpu
    {
        template<typename T>
        void Hadamard(const CudaRuntime& cudart, const GPUTensor<T>& vecin, GPUTensor<T>& vecinout);
        template<typename T>
        void Hadamard(const CudaRuntime& cudart, const GPUTensor<T>& vecin, const GPUTensor<T>& vecin2, GPUTensor<T>& vecout);
        
    } // namespace gpu
    
}   
#endif