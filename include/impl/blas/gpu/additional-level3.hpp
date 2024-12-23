#ifndef TCGMTENSOR_ADD_LEVEL3_HPP
#define TCGMTENSOR_ADD_LEVEL3_HPP
#include "linalg.hpp"
#include "runtime.hpp"

namespace tcgmtensor{
    namespace gpu
    {
        void MPSymMatrixMatrixMultiplication(const CudaRuntime& cudart, const Vector_<double>& d1, const Matrix_<float>& m1,
                                        const Vector_<double>& d2, const Matrix_<float>& m2, Vector_<double>& dout, Matrix_<float>& mout);
    } // namespace gpu
    
}   
#endif