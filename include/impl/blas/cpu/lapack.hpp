/// @file lapack.hpp
/// @brief C++-style LAPACK operations for matrix decompositions and solvers.
///
/// Provides template-based wrapper declarations for CPU LAPACK kernels operating on Matrix objects.
/// LAPACK operations include matrix factorizations (LU, QR, Cholesky), linear system solvers,
/// and eigenvalue decompositions. Each function is provided for double, float, and complex precision types.

#pragma once
#include "const.h"
#include "linalg.hpp"

namespace lahva
{

    namespace cpu
    {
        template<typename T>
        void SolveGenSysLinEquations(const char* Ta, Matrix_<T>& a, Matrix_<T>& b);
        template<typename T>
        void SolveGenSysLinEquations(Matrix_<T>& a, Matrix_<T>& b, const char* Ta = "N");
        template<typename T>
        void SolvePosSysLinEquations(Matrix_<T>& a, Matrix_<T>& b);
        template<typename T>
        void SolveGenSysLinEquations(const char* Ta, LowTriMatrix_<T>& a, Matrix_<T>& b);
        template<typename T>
        void SolveGenSysLinEquations(LowTriMatrix_<T>& a, Matrix_<T>& b, const char* Ta = "N");
        template<typename T>
        void SolveSymSysLinEquations(LowTriMatrix_<T>& a, Matrix_<T>& b);

        template<typename T>
        void InvertTriMatrix(Matrix_<T>&a);

        template <typename T>
        void InvertTriMatrix(Matrix_<T> &a);

        template <typename T>
        void SymEigenvalueDecomposition(Matrix_<T> &a, Vector_<T> &eigenvalues, char l_jobz = 'N');

        template <typename... Args>
        void SolveGenSysLinEquations(const CPURuntime &cudart_, Args &&...args)
        {
            (SolveGenSysLinEquations(args...));
        }

        template <typename... Args>
        void SolveSymSysLinEquations(const CPURuntime &cudart_, Args &&...args)
        {
            (SolveSymSysLinEquations(args...));
        }

        template <typename... Args>
        void SymEigenvalueDecomposition(const CPURuntime &cudart_, Args &&...args)
        {
            (SymEigenvalueDecomposition(args...));
        }

        template<typename... Args>
        void SolvePosSysLinEquations(const CPURuntime& cudart_, Args&&... args)
        {
            (SolvePosSysLinEquations(args...));
        }
        
    } // namespace cpu

}