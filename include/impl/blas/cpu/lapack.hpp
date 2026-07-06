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
        /// @brief Wrapper function to solve a general system of linear equations A*X = B, using LAPACK routines <T>getrf and <T>getrs.
        ///
        /// Solves the general linear system A*X = B using LU factorization with partial pivoting (getrf)
        /// followed by triangular solve (getrs). Matrix A is overwritten with its LU factorization.
        ///
        /// @tparam T Numerical element type (double, float, complex_double, complex_float).
        /// @param Ta Transpose option for matrix A: "N" (no transpose, A), "T" (transpose, A^T). Default: "N".
        /// @param a Input matrix A (overwritten with LU factorization).
        /// @param b Input-output matrix B (right-hand side, overwritten with solution X).
        template <typename T>
        void SolveGenSysLinEquations(const char *Ta, Matrix_<T> &a, Matrix_<T> &b);

        /// @brief Wrapper function to solve a general system of linear equations A*X = B with defaults, using LAPACK routines <T>getrf and <T>getrs.
        ///
        /// Solves the general linear system A*X = B using LU factorization with partial pivoting (getrf)
        /// followed by triangular solve (getrs). Matrix A is overwritten with its LU factorization.
        ///
        /// @tparam T Numerical element type (double, float, complex_double, complex_float).
        /// @param a Input matrix A (overwritten with LU factorization).
        /// @param b Input-output matrix B (right-hand side, overwritten with solution X).
        /// @param Ta Transpose option for matrix A: "N" (no transpose, A), "T" (transpose, A^T). Default: "N".
        template <typename T>
        void SolveGenSysLinEquations(Matrix_<T> &a, Matrix_<T> &b, const char *Ta = "N");

        /// @brief Wrapper function to solve a positive definite system of linear equations A*X = B, using LAPACK routines <T>potrf and <T>potrs.
        ///
        /// Solves the symmetric positive definite linear system A*X = B using Cholesky factorization (potrf)
        /// followed by triangular solve (potrs). Matrix A is assumed to be symmetric and positive definite.
        /// Matrix A is overwritten with its Cholesky factorization.
        ///
        /// @tparam T Numerical element type (double, float, complex_double, complex_float).
        /// @param a Input symmetric positive definite matrix A (overwritten with Cholesky factorization).
        /// @param b Input-output matrix B (right-hand side, overwritten with solution X).
        template <typename T>
        void SolvePosSysLinEquations(Matrix_<T> &a, Matrix_<T> &b);

        /// @brief Wrapper function to solve a triangular system of linear equations T*X = B, using LAPACK routines <T>tptrs.
        ///
        /// Solves the triangular system T*X = B where T is a lower triangular matrix stored in packed format.
        /// The system can be solved as-is or with T transposed, depending on the transpose flag.
        ///
        /// @tparam T Numerical element type (double, float, complex_double, complex_float).
        /// @param Ta Transpose option for matrix T: "N" (no transpose, T), "T" (transpose, T^T).
        /// @param a Input lower triangular matrix T in packed format.
        /// @param b Input-output matrix B (right-hand side, overwritten with solution X).
        template <typename T>
        void SolveGenSysLinEquations(const char *Ta, LowTriMatrix_<T> &a, Matrix_<T> &b);

        /// @brief Wrapper function to solve a triangular system of linear equations T*X = B with defaults, using LAPACK routines <T>tptrs.
        ///
        /// Solves the triangular system T*X = B where T is a lower triangular matrix stored in packed format.
        /// Uses default transpose option "N" (no transpose).
        ///
        /// @tparam T Numerical element type (double, float, complex_double, complex_float).
        /// @param a Input lower triangular matrix T in packed format.
        /// @param b Input-output matrix B (right-hand side, overwritten with solution X).
        /// @param Ta Transpose option for matrix T: "N" (no transpose, T), "T" (transpose, T^T). Default: "N".
        template <typename T>
        void SolveGenSysLinEquations(LowTriMatrix_<T> &a, Matrix_<T> &b, const char *Ta = "N");

        /// @brief Wrapper function to solve a symmetric system of linear equations A*X = B, using LAPACK routines <T>sptrs.
        ///
        /// Solves the symmetric linear system A*X = B where A is a symmetric matrix stored in packed format.
        /// Assumes A has already been factorized via <T>sptrf.
        ///
        /// @tparam T Numerical element type (double, float, complex_double, complex_float).
        /// @param a Input symmetric matrix A in packed format (previously factorized).
        /// @param b Input-output matrix B (right-hand side, overwritten with solution X).
        template <typename T>
        void SolveSymSysLinEquations(LowTriMatrix_<T> &a, Matrix_<T> &b);

        /// @brief Wrapper function to invert a triangular matrix, using LAPACK routines <T>trtri.
        ///
        /// Computes the inverse of a lower triangular matrix A. The inverse is stored back in A,
        /// overwriting the original matrix.
        ///
        /// @tparam T Numerical element type (double, float, complex_double, complex_float).
        /// @param a Input lower triangular matrix A (overwritten with its inverse).
        template <typename T>
        void InvertTriMatrix(Matrix_<T> &a);

        /// @brief Wrapper function to compute the eigenvalue decomposition of a symmetric matrix, using LAPACK routines <T>syevd.
        ///
        /// Computes the eigenvalues and optionally the eigenvectors of a symmetric matrix A.
        /// The eigenvalues are computed in ascending order. Matrix A is overwritten with the eigenvectors
        /// if requested, and the eigenvalues are stored in a vector.
        ///
        /// @tparam T Numerical element type (double, float, complex_double, complex_float).
        /// @param a Input symmetric matrix A (overwritten with eigenvectors if l_jobz='V').
        /// @param eigenvalues Output vector to store the eigenvalues in ascending order.
        /// @param l_jobz Flag to control computation: 'V' to compute both eigenvalues and eigenvectors, 'N' to compute only eigenvalues. Default: 'N'.
        template <typename T>
        void SymEigenvalueDecomposition(Matrix_<T> &a, Vector_<T> &eigenvalues, char l_jobz = 'N');

        /// @brief Runtime-dispatching overload for symmetric multiply (CPURuntime first).
        ///
        /// The runtime parameter is ignored for CPU implementations but this overload
        /// mirrors the signatures used for GPU/CUDA dispatching.
        template <typename... Args>
        void SolveGenSysLinEquations(const CPURuntime &rt_, Args &&...args)
        {
            (SolveGenSysLinEquations(args...));
        }

        /// @brief Runtime-dispatching overload for symmetric multiply (CPURuntime first).
        ///
        /// The runtime parameter is ignored for CPU implementations but this overload
        /// mirrors the signatures used for GPU/CUDA dispatching.
        template <typename... Args>
        void SolveSymSysLinEquations(const CPURuntime &rt_, Args &&...args)
        {
            (SolveSymSysLinEquations(args...));
        }

        /// @brief Runtime-dispatching overload for symmetric multiply (CPURuntime first).
        ///
        /// The runtime parameter is ignored for CPU implementations but this overload
        /// mirrors the signatures used for GPU/CUDA dispatching.
        template <typename... Args>
        void SymEigenvalueDecomposition(const CPURuntime &rt_, Args &&...args)
        {
            (SymEigenvalueDecomposition(args...));
        }

        /// @brief Runtime-dispatching overload for symmetric multiply (CPURuntime first).
        ///
        /// The runtime parameter is ignored for CPU implementations but this overload
        /// mirrors the signatures used for GPU/CUDA dispatching.
        template <typename... Args>
        void SolvePosSysLinEquations(const CPURuntime &rt_, Args &&...args)
        {
            (SolvePosSysLinEquations(args...));
        }

    } // namespace cpu

}
