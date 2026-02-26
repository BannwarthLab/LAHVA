#include "const.h"
namespace lahva
{

    namespace cpu
    {
        /// @brief Wrapper function to solve a general system of linear equations A*X = B, using dgetrf and dgetrs from LAPACK
        /// @param T wether to transpose the matrix A ("T") or not ("N")
        /// @param n dimension of the square matrix A
        /// @param a pointer to the matrix A, stored in column-major order, of size n x n
        /// @param nrhs number of right-hand sides, i.e., the number of columns of the matrix B
        /// @param b pointer to the matrix B, stored in column-major order, of size n x nrhs
        void SolveGenSysLinEquations(const char *T, const int n, double *a, const int nrhs, double *b);

        /// @brief Wrapper function to solve a general system of linear equations A*X = B, using dgetrf and dgetrs from LAPACK
        /// @param n dimension of the square matrix A
        /// @param a pointer to the matrix A, stored in column-major order, of size n x n
        /// @param nrhs number of right-hand sides, i.e., the number of columns of the matrix B
        /// @param b pointer to the matrix B, stored in column-major order, of size n x nrhs
        /// @param T wether to transpose the matrix A ("T") or not ("N"), default value is "N"
        void SolveGenSysLinEquations(const int n, double *a, const int nrhs, double *b, const char *T);

        /// @brief Wrapper function to solve a general system of linear equations A*X = B, using sgetrf and sgetrs from LAPACK
        /// @param T wether to transpose the matrix A ("T") or not ("N")
        /// @param n dimension of the square matrix A
        /// @param a pointer to the matrix A, stored in column-major order, of size n x n
        /// @param nrhs number of right-hand sides, i.e., the number of columns of the matrix B
        /// @param b pointer to the matrix B, stored in column-major order, of size n x nrhs
        void SolveGenSysLinEquations(const char *T, const int n, float *a, const int nrhs, float *b);

        /// @brief Wrapper function to solve a general system of linear equations A*X = B, using sgetrf and sgetrs from LAPACK
        /// @param n dimension of the square matrix A
        /// @param a pointer to the matrix A, stored in column-major order, of size n x n
        /// @param nrhs number of right-hand sides, i.e., the number of columns of the matrix B
        /// @param b pointer to the matrix B, stored in column-major order, of size n x nrhs
        /// @param T wether to transpose the matrix A ("T") or not ("N"), default value is "N"
        void SolveGenSysLinEquations(const int n, float *a, const int nrhs, float *b, const char *T);
    } // namespace cpu

}