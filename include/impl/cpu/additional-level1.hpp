#ifndef TCGMTENSOR_ADD_LEVEL1_CPU_HPP
#define TCGMTENSOR_ADD_LEVEL1_CPU_HPP
#include "linalg.hpp"
#include "const.h"

namespace tcgmtensor{
    namespace cpu
    {
        
        template<typename T>
        double ComputeTrace(const CPURuntime& cudart, const Matrix<T>&, bool use_diag = false);
        double ComputeTrace(const CPURuntime& cudart, const Vector<double>& diag);

        template<typename T>
        T FrobeniusNorm(const Tensor<T>& mat);
        template<typename T>
        T FrobeniusNorm(const CPURuntime& cudart, const Tensor<T>& mat);

        template<typename T>
        T FrobeniusInnerProduct(const Matrix<T>& mat1, const Matrix<T>& mat2);
        template<typename T>
        T FrobeniusInnerProduct(const CPURuntime& rt, const Matrix<T>& mat1, const Matrix<T>& mat2);

        template<typename T>
        void GetDiagonal(const CPURuntime& cudart, const Matrix<T>& mat, Vector<T>& vec);

        template<typename T>
        void SetDiagonal(const CPURuntime& cudart, const Vector<T>& vec, Matrix<T>& m);  
    } // namespace gpu
    
}   
#endif