#include "impl/gpu/additional-level1.hpp"
#include "impl/gpu/level3.hpp"
#include "impl/gpu/additional-level3.hpp"
#include "custom-kernel/mixed-precision.h"
#include "../../gpu-utils/utils.hpp"

namespace tcgmtensor{
    namespace gpu
    {
        
        void MPSymMatrixMatrixMultiplication(const CudaRuntime& cudart, const Vector<double>& d1, const Matrix<float>& m1,
                                        const Vector<double>& d2, const Matrix<float>& m2, Vector<double>& dout, Matrix<float>& mout)
        {
            SymMatrixMatrixProduct(cudart, m1, m2, mout);
            Hadamard(cudart, d1, d2, dout);
            SymmetrizedON2ScalingProductGPU(cudart, d1, m1, d2, m2, mout);
            SymmetrizeMatrix(cudart, mout);
        }

    } // namespace gpu
}   