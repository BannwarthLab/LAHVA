/// @file lapack.h
/// @brief C-style LAPACK operations for matrix decompositions and solvers.
///
/// Provides thin wrapper declarations for CPU LAPACK kernels operating on raw pointers.
/// LAPACK operations include matrix factorizations (LU, QR, Cholesky), linear system solvers,
/// and eigenvalue decompositions. Each function is provided for double and float precision.

#include "const.h"

namespace lahva
{

    namespace cpu
    {
        /// @brief Solve a general system of linear equations A*X = B (double precision)
        ///
        /// Solves the general linear system A*X = B using LU factorization with partial pivoting (dgetrf)
        /// followed by triangular solve (dgetrs). Matrix A is overwritten with its LU factorization.
        ///
        /// @param Ta Transpose option for matrix A: "N" (no transpose, A*X = B), "T" (transpose, A^T*X = B).
        /// @param n Dimension of the square matrix A (n x n).
        /// @param a Pointer to matrix A (double precision, n x n, column-major, overwritten with LU factorization).
        /// @param nrhs Number of right-hand sides (number of columns in B).
        /// @param b Pointer to matrix B (double precision, n x nrhs, column-major, overwritten with solution X).
        void SolveGenSysLinEquations(const char *Ta, const int n, double *a, const int nrhs, double *b);

        /// @brief Solve a general system of linear equations with default transpose (double precision)
        ///
        /// Convenience overload with default transpose option.
        /// Solves the general linear system A*X = B using LU factorization (dgetrf) and triangular solve (dgetrs).
        ///
        /// @param n Dimension of the square matrix A (n x n).
        /// @param a Pointer to matrix A (double precision, n x n, column-major, overwritten with LU factorization).
        /// @param nrhs Number of right-hand sides (number of columns in B).
        /// @param b Pointer to matrix B (double precision, n x nrhs, column-major, overwritten with solution X).
        /// @param Ta Transpose option for matrix A: "N" (no transpose, A*X = B), "T" (transpose, A^T*X = B). Default: "N".
        void SolveGenSysLinEquations(const int n, double *a, const int nrhs, double *b, const char *Ta);

        /// @brief Solve a general system of linear equations A*X = B (single precision)
        ///
        /// Solves the general linear system A*X = B using LU factorization with partial pivoting (sgetrf)
        /// followed by triangular solve (sgetrs). Matrix A is overwritten with its LU factorization.
        ///
        /// @param Ta Transpose option for matrix A: "N" (no transpose, A*X = B), "T" (transpose, A^T*X = B).
        /// @param n Dimension of the square matrix A (n x n).
        /// @param a Pointer to matrix A (single precision, n x n, column-major, overwritten with LU factorization).
        /// @param nrhs Number of right-hand sides (number of columns in B).
        /// @param b Pointer to matrix B (single precision, n x nrhs, column-major, overwritten with solution X).
        void SolveGenSysLinEquations(const char *Ta, const int n, float *a, const int nrhs, float *b);

        /// @brief Solve a general system of linear equations with default transpose (single precision)
        ///
        /// Convenience overload with default transpose option.
        /// Solves the general linear system A*X = B using LU factorization (sgetrf) and triangular solve (sgetrs).
        ///
        /// @param n Dimension of the square matrix A (n x n).
        /// @param a Pointer to matrix A (single precision, n x n, column-major, overwritten with LU factorization).
        /// @param nrhs Number of right-hand sides (number of columns in B).
        /// @param b Pointer to matrix B (single precision, n x nrhs, column-major, overwritten with solution X).
        /// @param Ta Transpose option for matrix A: "N" (no transpose, A*X = B), "T" (transpose, A^T*X = B). Default: "N".
        void SolveGenSysLinEquations(const int n, float *a, const int nrhs, float *b, const char *Ta);
    } // namespace cpu

}