#include "const.h"
#include "linalg.hpp"

namespace tcgmtensor{

    namespace cpu
    {
        template<typename T>
        void SolveGenSysLinEquations(const char* Ta, Matrix<T>& a, Matrix<T>& b);
        template<typename T>
        void SolveGenSysLinEquations(Matrix<T>& a, Matrix<T>& b, const char* Ta = "N");

        template<typename... Args>
        void SolveGenSysLinEquations(const CPURuntime& cudart_, Args&&... args)
        {
            (SolveGenSysLinEquations(args...));
        }

        
        
    } // namespace cpu
    
}