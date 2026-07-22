/// @file additional-level3.hpp
/// @brief GPU-accelerated additional Level-3 operations beyond standard BLAS.
///
/// Provides GPU kernel declarations for supplementary matrix-matrix operations
/// operating on GPU matrix objects. These extend standard BLAS Level-3 functionality.
/// Each function is provided for double and float precision with GPU runtime management.

#pragma once
#include "linalg.hpp"
#include "runtime.hpp"
#include "impl/blas/gpu/additional-level1.hpp"
#include "impl/blas/gpu/level3.hpp"
#include "impl/blas/gpu/level1.hpp"
#include "impl/blas/gpu/additional-level2.hpp"
#include "additional-level3-add.hpp"
namespace lahva
{
    namespace gpu
    {

        template<typename Allocator, typename GPUAllocator, typename All2, typename GPUAll2>
        void MPSymMatrixMatrixMultiplication(const CudaRuntime& cudart, const Vector<double, Allocator, GPUAllocator>& d1, const Matrix<float, All2, GPUAll2>& m1,
                                             const Vector<double, Allocator, GPUAllocator>& d2, const Matrix<float, All2, GPUAll2>& m2, 
                                             Vector<double, Allocator, GPUAllocator>& dout, Matrix<float, All2, GPUAll2>& mout, bool fast = false)
        {
            fast = true;
            MatrixMatrixProduct(cudart, m1, m2, mout);
            HadamardProduct(cudart, d1, d2, dout);
            SymmetrizedON2ScalingProductGPU(cudart, d1, m1, d2, m2, mout);
            SymmetrizeMatrix(cudart, mout);
            
        };

        template<typename Allocator, typename GPUAllocator, typename All2, typename GPUAll2>
        void MPSymMatrixMatrixMultiplication(const CudaRuntime& cudart, const MPRuntime& mp_rt, const Vector<double, Allocator, GPUAllocator>& d1, const MixedPrecisionMatrix<float, All2, GPUAll2>& m1,
                                             const Vector<double, Allocator, GPUAllocator>& d2, const MixedPrecisionMatrix<float, All2, GPUAll2>& m2, 
                                             Vector<double, Allocator, GPUAllocator>& dout, Matrix<float, All2, GPUAll2>& mout)
        {
            
            if (mp_rt.fast_mode)
            {
                //std::cout << "Using mixed precision matrix multiplication" << std::endl;
                m1.resetSplit();
                m2.resetSplit();
                SymMatrixMatrixProduct(cudart, mp_rt, m1, m2, mout);
            }
            else
            {
                //std::cout << "Using full precision matrix multiplication" << std::endl;
                MatrixMatrixProduct(cudart, m1, m2, mout);
            }
            HadamardProduct(cudart, d1, d2, dout);
            SymmetrizedON2ScalingProductGPU(cudart, d1, m1, d2, m2, mout);
            SymmetrizeMatrix(cudart, mout);
            
        };
    

       
    } // namespace gpu
    
}