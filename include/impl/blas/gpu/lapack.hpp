/// @file lapack.hpp
/// @brief GPU-accelerated LAPACK operations
///
/// Provides GPU LAPACK-style kernel declarations for CUDA acceleration, implemented with
/// NVIDIA's cuSOLVER dense library. Each function is provided for double and float precision
/// with GPU runtime management.

#pragma once
#include "const.h"
#include "linalg.hpp"
#include "runtime.hpp"

namespace lahva
{
    namespace gpu
    {

    /// @brief Computes the inverse of a general square matrix, using cuSOLVER functions cusolverDnXgetrf and cusolverDnXgetrs (CUDA_R_64F).
    ///
    /// Computes the LU factorization with partial pivoting of A (getrf) and solves A*X = I (getrs),
    /// so that X = A^(-1). The inverse is stored back in A on the device, overwriting the original
    /// matrix. 
    ///
    /// @param cudart CUDA runtime instance.
    /// @param a Input/output square GPU matrix A (Matrix_<double>), overwritten with its inverse.
    /// @throw std::runtime_error if A is not square, or if A is singular (getrf/getrs fails).
    template <typename T>
    void InvertMatrix(CudaRuntime& cudart, Matrix_<T>& a);

    /// @brief Solves a general system of linear equations op(A)*X = B, using cuSOLVER functions cusolverDnXgetrf and cusolverDnXgetrs.
    ///
    /// Computes the LU factorization with partial pivoting of A (getrf) followed by forward/backward
    /// substitution (getrs). A is overwritten on the device with its LU factors and B with the solution X.
    ///
    /// @tparam T Floating-point type (double or float).
    /// @param cudart CUDA runtime instance.
    /// @param Ta Operation on matrix A: "N" (no transpose, A*X = B), "T" (transpose, A^T*X = B).
    /// @param a Input square GPU matrix A (n x n), overwritten with its LU factorization.
    /// @param b Input/output GPU matrix B (n x nrhs), overwritten with the solution X.
    /// @throw std::runtime_error if A is not square, B does not have n rows, or A is singular (getrf/getrs fails).
    template <typename T>
    void SolveGenSysLinEquations(CudaRuntime& cudart, const char* Ta, Matrix_<T>& a, Matrix_<T>& b);

    /// @brief Solves a general system of linear equations op(A)*X = B with defaults, using cuSOLVER functions cusolverDnXgetrf and cusolverDnXgetrs.
    ///
    /// Convenience overload with the operation on A as the last parameter (default: "N").
    /// A is overwritten on the device with its LU factors and B with the solution X.
    ///
    /// @tparam T Floating-point type (double or float).
    /// @param cudart CUDA runtime instance.
    /// @param a Input square GPU matrix A (n x n), overwritten with its LU factorization.
    /// @param b Input/output GPU matrix B (n x nrhs), overwritten with the solution X.
    /// @param Ta Operation on matrix A: "N" (no transpose, A*X = B), "T" (transpose, A^T*X = B). Default: "N".
    /// @throw std::runtime_error if A is not square, B does not have n rows, or A is singular (getrf/getrs fails).
    template <typename T>
    void SolveGenSysLinEquations(CudaRuntime& cudart, Matrix_<T>& a, Matrix_<T>& b, const char* Ta = "N");

    } // namespace gpu
} // namespace lahva
