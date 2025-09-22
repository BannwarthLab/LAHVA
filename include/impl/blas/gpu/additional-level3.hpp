#ifndef LAHVA_ADD_LEVEL3_HPP
#define LAHVA_ADD_LEVEL3_HPP
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
        void MPSymMatrixMatrixMultiplication(const CudaRuntime& cudart, const Vector<double, Allocator, GPUAllocator>& d1, const MixedPrecisionMatrix<float, All2, GPUAll2>& m1,
                                             const Vector<double, Allocator, GPUAllocator>& d2, const MixedPrecisionMatrix<float, All2, GPUAll2>& m2, 
                                             Vector<double, Allocator, GPUAllocator>& dout, Matrix<float, All2, GPUAll2>& mout, Matrix<float, All2, GPUAll2>& buffer)
        {
            
            if (cudart.fast_mp)
            {
                //std::cout << "Using mixed precision matrix multiplication" << std::endl;
                m1.resetSplit();
                m2.resetSplit();
                SymMatrixMatrixProduct(cudart, m1, m2, mout, buffer);
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
#endif