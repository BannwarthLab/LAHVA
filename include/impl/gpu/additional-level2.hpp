#ifndef TCGMTENSOR_ADD_LEVEL2_HPP
#define TCGMTENSOR_ADD_LEVEL2_HPP
#include "linalg.hpp"
#include "runtime.hpp"

namespace tcgmtensor{
    namespace gpu
    {
       template<typename T>
       void PackedTri2TriMatrix(const CudaRuntime&, const LowTriMatrix<T>& PackedTriMatrix, Matrix<T>& TriMatrix);
       template<typename T>
       void TriMatrix2PackedTri(const CudaRuntime&, const Matrix<T>& TriMatrix, LowTriMatrix<T>& PackedTriMatrix);
    } // namespace gpu
    
}   
#endif