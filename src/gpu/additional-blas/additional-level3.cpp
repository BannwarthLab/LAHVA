#include "impl/blas/gpu/additional-level1.hpp"
#include "impl/blas/gpu/level3.hpp"
#include "impl/blas/gpu/additional-level3.hpp"
#include "custom-kernel/mixed-precision.h"
#include "../../gpu-utils/utils.hpp"
#include "timer.hpp"
#include "impl/blas/gpu/additional-level1.hpp"
#include "impl/blas/gpu/level3.hpp"
namespace tcgmtensor{
    namespace gpu
    {
        
        void MPSymMatrixMatrixMultiplication(const CudaRuntime& cudart, const Vector_<double>& d1, const Matrix_<float>& m1,
                                        const Vector_<double>& d2, const Matrix_<float>& m2, Vector_<double>& dout, Matrix_<float>& mout, bool fast)
        {
            fast = true;
            if (fast)
            {
                //std::cout << "Fast" << std::endl;
                MatrixMatrixProductTF32(cudart, m1, m2, mout, 1.0, 0.0);
            }
            else
            {
                MatrixMatrixProduct(cudart, m1, m2, mout, 1.0, 0.0);
                //SymMatrixMatrixProduct(cudart, m1, m2, mout);
            }
            SymmetrizeMatrix(cudart, mout);
            Hadamard(cudart, d1, d2, dout);
            SymmetrizedON2ScalingProductGPU(cudart, d1, m1, d2, m2, mout);
            SymmetrizeMatrix(cudart, mout);
        }

        

    } // namespace gpu
}   