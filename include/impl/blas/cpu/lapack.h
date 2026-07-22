/// @file lapack.h
/// @brief C-style LAPACK operations for matrix decompositions and solvers.
///
/// Provides thin wrapper declarations for CPU LAPACK kernels operating on raw pointers.
/// LAPACK operations include matrix factorizations (LU, QR, Cholesky), linear system solvers,
/// and eigenvalue decompositions. Each function is provided for double and float precision.

#pragma once
#include "const.h"
namespace lahva{

    namespace cpu
    {
        void SolveGenSysLinEquations(const char* T, const int n, double* a, const int nrhs, double* b);
        void SolveGenSysLinEquations(const int n, double* a, const int nrhs, double* b, const char* T);
        void SolveGenSysLinEquations(const char* T, const int n, float* a, const int nrhs, float* b);
        void SolveGenSysLinEquations(const int n, float* a, const int nrhs, float* b, const char* T);
    } // namespace cpu

}