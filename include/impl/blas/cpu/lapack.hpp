#include "const.h"
#include "linalg.hpp"

namespace lahva
{

    namespace cpu
    {
        /// @brief Wrapper function to solve a general system of linear equations A*X = B, using LAPACK routines <T>getrf and <T>getrs
        /// @tparam T Numerical type of the matrices
        /// @param Ta wether to transpose the matrix A ("T") or not ("N")
        /// @param a Matrix A
        /// @param b Matrix B
        template <typename T>
        void SolveGenSysLinEquations(const char *Ta, Matrix_<T> &a, Matrix_<T> &b);

        /// @brief Wrapper function to solve a general system of linear equations A*X = B, using LAPACK routines <T>getrf and <T>getrs
        /// @tparam T Numerical type of the matrices
        /// @param a Matrix A
        /// @param b Matrix B
        /// @param Ta wether to transpose the matrix A ("T") or not ("N"), default value is "N"
        template <typename T>
        void SolveGenSysLinEquations(Matrix_<T> &a, Matrix_<T> &b, const char *Ta = "N");

        /// @brief Wrapper function to solve a positive definite system of linear equations A*X = B, using LAPACK routines <T>potrf and <T>potrs
        /// @tparam T Numerical type of the matrices
        /// @param a Matrix A
        /// @param b Matrix B
        template <typename T>
        void SolvePosSysLinEquations(Matrix_<T> &a, Matrix_<T> &b);

        /// @brief Wrapper function to solve a symmetric system of linear equations A*X = B, using LAPACK routines <T>tptrs
        /// @tparam T Numerical type of the matrices
        /// @param Ta wether to transpose the matrix A ("T") or not ("N")
        /// @param a Lower Triangular Matrix A (packed format)
        /// @param b Matrix B
        template <typename T>
        void SolveGenSysLinEquations(const char *Ta, LowTriMatrix_<T> &a, Matrix_<T> &b);

        /// @brief Wrapper function to solve a symmetric system of linear equations A*X = B, using LAPACK routines <T>tptrs
        /// @tparam T Numerical type of the matrices
        /// @param a Lower Triangular Matrix A (packed format)
        /// @param b Matrix B
        /// @param Ta wether to transpose the matrix A ("T") or not ("N"), default value is "N"
        template <typename T>
        void SolveGenSysLinEquations(LowTriMatrix_<T> &a, Matrix_<T> &b, const char *Ta = "N");

        /// @brief Wrapper function to solve a symmetric system of linear equations A*X = B, using LAPACK routines <T>sptrs
        /// @tparam T Numerical type of the matrices
        /// @param a Lower Triangular Matrix A (packed format)
        /// @param b Matrix B
        template <typename T>
        void SolveSymSysLinEquations(LowTriMatrix_<T> &a, Matrix_<T> &b);

        /// @brief Wrapper function to invert a triangular matrix, using LAPACK routines <T>trtri
        /// @tparam T Numerical type of the matrix
        /// @param a Triangular Matrix A, default is lower triangular
        template <typename T>
        void InvertTriMatrix(Matrix_<T> &a);

        /// @brief Wrapper function to compute the eigenvalue decomposition of a symmetric matrix, using LAPACK routines <T>syevd
        /// @tparam T Numerical type of the matrix
        /// @param a Symmetric Matrix A
        /// @param eigenvalues Vector to store the eigenvalues
        /// @param l_jobz 'V' to compute eigenvectors, 'N' to compute only eigenvalues, default is 'N'
        template <typename T>
        void SymEigenvalueDecomposition(Matrix_<T> &a, Vector_<T> &eigenvalues, char l_jobz = 'N');

        /// @brief Wrapper function to solve a general system of linear equations A*X = B, using LAPACK routines <T>getrf and <T>getrs, with CPURuntime as first argument
        /// @tparam ...Args additional input arguments forwarded to SolveGenSysLinEquations
        /// @param rt_ CPURuntime instance
        /// @param ...args additional input arguments forwarded to SolveGenSysLinEquations
        template <typename... Args>
        void SolveGenSysLinEquations(const CPURuntime &rt_, Args &&...args)
        {
            (SolveGenSysLinEquations(args...));
        }

        /// @brief Wrapper function to solve a symmetric system of linear equations A*X = B, using LAPACK routines <T>sptrs, with CPURuntime as first argument
        /// @tparam ...Args additional input arguments forwarded to SolveSymSysLinEquations
        /// @param rt_ CPURuntime instance
        /// @param ...args additional input arguments forwarded to SolveSymSysLinEquations
        template <typename... Args>
        void SolveSymSysLinEquations(const CPURuntime &rt_, Args &&...args)
        {
            (SolveSymSysLinEquations(args...));
        }

        /// @brief Wrapper function to compute the eigenvalue decomposition of a symmetric matrix, using LAPACK routines <T>syevd, with CPURuntime as first argument
        /// @tparam ...Args additional input arguments forwarded to SymEigenvalueDecomposition
        /// @param rt_ CPURuntime instance
        /// @param ...args additional input arguments forwarded to SymEigenvalueDecomposition
        template <typename... Args>
        void SymEigenvalueDecomposition(const CPURuntime &rt_, Args &&...args)
        {
            (SymEigenvalueDecomposition(args...));
        }

        /// @brief Wrapper function to solve a positive definite system of linear equations A*X = B, using LAPACK routines <T>potrf and <T>potrs, with CPURuntime as first argument
        /// @tparam ...Args additional input arguments forwarded to SolvePosSysLinEquations
        /// @param rt_ CPURuntime instance
        /// @param ...args additional input arguments forwarded to SolvePosSysLinEquations
        template <typename... Args>
        void SolvePosSysLinEquations(const CPURuntime &rt_, Args &&...args)
        {
            (SolvePosSysLinEquations(args...));
        }

    } // namespace cpu

}