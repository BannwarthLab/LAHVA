/// @file lapack.hpp
/// @brief GPU-accelerated LAPACK operations for matrix decompositions and solvers.
///
/// Provides GPU LAPACK kernel declarations for CUDA acceleration via cuSOLVER.
/// LAPACK operations include matrix factorizations (LU, QR, Cholesky), linear system solvers,
/// and eigenvalue decompositions. Each function is provided for double and float precision.

#include "const.h"
#include "runtime.hpp"
#include "linalg.hpp"
#include "../lapack.hpp"

namespace lahva{

    namespace gpu
    {
        //template<typename T, typename Allocator, typename GPUAllocator>
        //void SolveGenSysLinEquations(CudaRuntime& cudart, const char* Ta, Matrix<T, Allocator, GPUAllocator>& a, Matrix<T, Allocator, GPUAllocator>& b);
        //template<typename T, typename Allocator, typename GPUAllocator>
        //void SolveGenSysLinEquations(CudaRuntime& cudart, Matrix<T, Allocator, GPUAllocator>& a, Matrix<T, Allocator, GPUAllocator>& b, const char* Ta = "N");
        
    } // namespace cpu
}