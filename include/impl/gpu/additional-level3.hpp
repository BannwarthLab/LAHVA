#ifndef TCGMTENSOR_ADD_LEVEL3_HPP
#define TCGMTENSOR_ADD_LEVEL3_HPP
#include "linalg.hpp"
#include "runtime.hpp"

namespace tcgmtensor{
    namespace gpu
    {
        void MPSymMatrixMatrixMultiplication(const CudaRuntime& cudart, const Vector<double>& d1, const Matrix<float>& m1,
                                        const Vector<double>& d2, const Matrix<float>& m2, Vector<double>& dout, Matrix<float>& mout);
    } // namespace gpu
    
}   
#endif