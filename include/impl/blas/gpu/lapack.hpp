#include "const.h"
#include "runtime.hpp"
#include "linalg.hpp"

namespace lahva{

    namespace gpu
    {
        template<typename T, typename Allocator, typename GPUAllocator>
        void SolveGenSysLinEquations(CudaRuntime& cudart, const char* Ta, Matrix<T, Allocator, GPUAllocator>& a, Matrix<T, Allocator, GPUAllocator>& b);
        template<typename T, typename Allocator, typename GPUAllocator>
        void SolveGenSysLinEquations(CudaRuntime& cudart, Matrix<T, Allocator, GPUAllocator>& a, Matrix<T, Allocator, GPUAllocator>& b, const char* Ta = "N");
        
    } // namespace cpu
}