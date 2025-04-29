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
            //SymmetrizeMatrix(cudart, mout);
            HadamardProduct(cudart, d1, d2, dout);
            SymmetrizedON2ScalingProductGPU(cudart, d1, m1, d2, m2, mout);
            SymmetrizeMatrix(cudart, mout);
            
        };

    

        template<typename Allocator, typename GPUAllocator>
        void MatrixMatrixProductFP32Approx(const CudaRuntime& cudart, const Matrix<double, Allocator, GPUAllocator>& A, const Matrix<double, Allocator, GPUAllocator>& B, Matrix<double, Allocator, GPUAllocator>& C,
        const double alpha = 1.0 , const double beta = 0.0, const char* Ta = "N", const char* Tb = "N")
        {
            using MyMatrix = Matrix<float> ;
            MyMatrix A1(A.shape(), A.get_allocator(), A.get_gpuallocator());
            MyMatrix A2(A.shape(), A.get_allocator(), A.get_gpuallocator());
            MyMatrix B1(B.shape(), cudart, B.get_gpuallocator());
            MyMatrix B2(B.shape(), cudart, B.get_gpuallocator());

            DecomposeVector2MP(cudart, A, A1, A2);
            A1.copy2host(cudart);
            A2.copy2host(cudart);
            cudart.synchronize();
            A1.print();
            std::cout << std::endl;
            A2.print();
            std::cout << std::endl;
            DecomposeVector2MP(cudart, B, B1, B2);
            
            MyMatrix C1(C.shape(), cudart, C.get_gpuallocator());
            
            ScaleVector(cudart, beta, C);

            //MatrixMatrixProduct(cudart, A1, B1, C1, float(alpha), 0.0, Ta, Tb);
            AddVectors(cudart, A1, C);
            //MatrixMatrixProduct(cudart, A2, B2, C1, float(alpha), 0.0, Ta, Tb);
            //AddVectors(cudart, C1, C);
            //MatrixMatrixProduct(cudart, A2, B1, C1, float(alpha), 0.0, Ta, Tb);
            //AddVectors(cudart, C1, C);
            //MatrixMatrixProduct(cudart, A1, B2, C1, float(alpha), 0.0, Ta, Tb);
            //AddVectors(cudart, C1, C);
            

        };
       
    } // namespace gpu
    
}   
#endif