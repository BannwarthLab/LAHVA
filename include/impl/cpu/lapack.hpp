#include "const.h"
#include "linalg.hpp"

namespace tcgmtensor{

    namespace cpu
    {
        template<typename T>
        void SolveGenSysLinEquations(const char* Ta, Matrix<T>& a, Matrix<T>& b);
        template<typename T>
        void SolveGenSysLinEquations(Matrix<T>& a, Matrix<T>& b, const char* Ta = "N");
        template<typename T>
        void SolveGenSysLinEquations(const char* Ta, LowTriMatrix<T>& a, Matrix<T>& b);
        template<typename T>
        void SolveGenSysLinEquations(LowTriMatrix<T>& a, Matrix<T>& b, const char* Ta = "N");
        template<typename T>
        void SolveSymSysLinEquations(LowTriMatrix<T>& a, Matrix<T>& b);

        template<typename... Args>
        void SolveGenSysLinEquations(const CPURuntime& cudart_, Args&&... args)
        {
            (SolveGenSysLinEquations(args...));
        }


        template<typename... Args>
        void SolveSymSysLinEquations(const CPURuntime& cudart_, Args&&... args)
        {
            (SolveSymSysLinEquations(args...));
        }


        
        
    } // namespace cpu
    
}