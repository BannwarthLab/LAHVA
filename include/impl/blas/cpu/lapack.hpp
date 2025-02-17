#include "const.h"
#include "linalg.hpp"

namespace lahva{

    namespace cpu
    {
        template<typename T>
        void SolveGenSysLinEquations(const char* Ta, Matrix_<T>& a, Matrix_<T>& b);
        template<typename T>
        void SolveGenSysLinEquations(Matrix_<T>& a, Matrix_<T>& b, const char* Ta = "N");
        template<typename T>
        void SolveGenSysLinEquations(const char* Ta, LowTriMatrix_<T>& a, Matrix_<T>& b);
        template<typename T>
        void SolveGenSysLinEquations(LowTriMatrix_<T>& a, Matrix_<T>& b, const char* Ta = "N");
        template<typename T>
        void SolveSymSysLinEquations(LowTriMatrix_<T>& a, Matrix_<T>& b);

        template<typename T>
        void InvertTriMatrix(Matrix_<T>&a);

        template <typename T>
        void SymEigenvalueDecomposition(Matrix_<T> &a, Vector_<T>& eigenvalues, char l_jobz = 'N');

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


        template<typename... Args>
        void SymEigenvalueDecomposition(const CPURuntime& cudart_, Args&&... args)
        {
            (SymEigenvalueDecomposition(args...));
        }
        
    } // namespace cpu
    
}