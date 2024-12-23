#ifndef TCGMTENSOR_ADD_LEVEL2_CPU_HPP
#define TCGMTENSOR_ADD_LEVEL2_CPU_HPP
#include "linalg.hpp"
#include "const.h"

namespace tcgmtensor{
    namespace cpu
    {
        
        template<typename T>
        void AddMatrices(const char* Ta, const char* Tb, const T alpha, const Matrix_<T>& a, const Matrix_<T>& b,
        const T beta, Matrix_<T>& c);
        template<typename T>
        void AddMatrices(const Matrix_<T>& a, const Matrix_<T>& b, Matrix_<T>& c,
        const T alpha = 1.0 , const T beta = 1.0, const char* Ta = "N", const char* Tb = "N");

        template<typename... Args>
        void AddMatrices(const CPURuntime& rt_, Args&&... args) {
        (AddMatrices(args...));                
        }

    } // namespace gpu
    
}   
#endif