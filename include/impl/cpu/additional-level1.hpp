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
        T FrobeniusNorm(const Matrix<T>& mat);
        template<typename T>
        T FrobeniusNorm(const CPURuntime& cudart, const Matrix<T>& mat);
        
    } // namespace gpu
    
}   
#endif