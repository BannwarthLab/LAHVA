/// @file lapack.cpp
/// @brief CPU C++ template implementations of LAPACK linear algebra routines.
///
/// Provides C++ template functions for LAPACK operations including LU factorization, QR
/// factorization, Cholesky decomposition, eigenvalue solvers, and matrix inversions.
/// Functions accept Matrix<T> and Vector<T> types and forward to Fortran LAPACK routines.

#include <stdexcept>
#include "impl/blas/cpu/lapack.hpp"
#include "lapack_wrap.hpp"
#include "linalg.hpp"

namespace lahva
{

    namespace cpu
    {

        /// @brief Compute LU factorization of a double-precision matrix
        ///
        /// Computes the LU factorization with partial pivoting (DGETRF) of matrix A.
        /// Matrix A is overwritten with its LU factors (L unit lower triangular, U upper triangular).
        /// The pivot indices are stored in ipiv.
        ///
        /// @param a Input matrix A (double precision, square, overwritten with LU factorization).
        /// @param ipiv Output pivot index vector (adjusted size if necessary).
        /// @throw std::runtime_error if A is not square or if DGETRF fails.
        void LUFactorization(Matrix_<double>& a, Vector<LPCK_INT>& ipiv)
        {
            LPCK_INT info = 0;
            if (a.shape().first != a.shape().second)
            {
                throw std::runtime_error("A should be symmetric");
            };
            
            LPCK_INT n = a.shape().first;
            if ((LPCK_INT)ipiv.size() != n)
            {
                ipiv = Vector<LPCK_INT>(n);
            }
#ifdef _APPLE
            dgetrf_(&n, &n, a.data(), &n, ipiv.data(), &info);
#elif defined(W_MKL)
            info = LAPACKE_dgetrf(l_major, n, n, a.data(), n, ipiv.data());
#else
            LAPACK_dgetrf(&n, &n, a.data(), &n, ipiv.data(), &info);
#endif
            if (info != 0)
            {
                throw std::runtime_error("Failure in DGETRF");
            }
        };

        /// @brief Compute LU factorization of a single-precision matrix
        ///
        /// Computes the LU factorization with partial pivoting (SGETRF) of matrix A.
        /// Matrix A is overwritten with its LU factors (L unit lower triangular, U upper triangular).
        /// The pivot indices are stored in ipiv.
        ///
        /// @param a Input matrix A (single precision, square, overwritten with LU factorization).
        /// @param ipiv Output pivot index vector (adjusted size if necessary).
        /// @throw std::runtime_error if A is not square or if SGETRF fails.
        void LUFactorization(Matrix_<float>& a, Vector<LPCK_INT>& ipiv)
        {
            LPCK_INT info = 0;
            if (a.shape().first != a.shape().second)
            {
                throw std::runtime_error("A should be symmetric");
            };
            
            LPCK_INT n = a.shape().first;
            if (static_cast<size_t>(ipiv.size()) != static_cast<size_t>(n))
            {
                ipiv = Vector<LPCK_INT>(n);
            }
#ifdef _APPLE
            sgetrf_(&n, &n, a.data(), &n, ipiv.data(), &info);
#elif defined(W_MKL)
            info = LAPACKE_sgetrf(l_major, n, n, a.data(), n, ipiv.data());
#else
            LAPACK_sgetrf(&n, &n, a.data(), &n, ipiv.data(), &info);
#endif
            if (info != 0)
            {
                throw std::runtime_error("Failure in DGETRF");
            }
        };

        /// @brief Compute Cholesky factorization of a double-precision symmetric positive definite matrix
        ///
        /// Computes the Cholesky factorization (DPOTRF) of matrix A.
        /// Matrix A is assumed to be symmetric and positive definite.
        /// A is overwritten with its Cholesky factor (upper or lower triangular).
        ///
        /// @param a Input symmetric positive definite matrix A (double precision, square, overwritten with Cholesky factor).
        /// @throw std::runtime_error if A is not square or if DPOTRF fails (e.g., not positive definite).
        void CholeskyFactorization(Matrix_<double>& a)
        {
            LPCK_INT info = 0;
            if (a.shape().first != a.shape().second)
            {
                throw std::runtime_error("A should be symmetric");
            };
            
            LPCK_INT n = a.shape().first;
#ifdef _APPLE
            dpotrf_(&l_uplo, &n, a.data(), &n, &info);
#elif defined(W_MKL)
            info = LAPACKE_dpotrf(l_major, l_uplo, n, a.data(), n);
#else
            LAPACK_dpotrf(&l_uplo, &n, a.data(), &n, &info);
#endif
            if (info != 0)
            {
                throw std::runtime_error("Failure in DPOTRF");
            }
        };

        /// @brief Compute Cholesky factorization of a single-precision symmetric positive definite matrix
        ///
        /// Computes the Cholesky factorization (SPOTRF) of matrix A.
        /// Matrix A is assumed to be symmetric and positive definite.
        /// A is overwritten with its Cholesky factor (upper or lower triangular).
        ///
        /// @param a Input symmetric positive definite matrix A (single precision, square, overwritten with Cholesky factor).
        /// @throw std::runtime_error if A is not square or if SPOTRF fails (e.g., not positive definite).
        void CholeskyFactorization(Matrix_<float>& a)
        {
            LPCK_INT info = 0;
            if (a.shape().first != a.shape().second)
            {
                throw std::runtime_error("A should be symmetric");
            };
            
            LPCK_INT n = a.shape().first;
#ifdef _APPLE
            spotrf_(&l_uplo, &n, a.data(), &n, &info);
#elif defined(W_MKL)
            info = LAPACKE_spotrf(l_major, l_uplo, n, a.data(), n);
#else
            LAPACK_spotrf(&l_uplo, &n, a.data(), &n, &info);
#endif
            if (info != 0)
            {
                throw std::runtime_error("Failure in DPOTRF");
            }
        };

        /// @brief Solve a symmetric positive definite system of linear equations A*X = B (double precision)
        ///
        /// Solves the symmetric positive definite linear system A*X = B using Cholesky factorization (DPOTRF)
        /// followed by triangular solve (DPOTRS). Matrix A is assumed to be symmetric and positive definite.
        /// Matrix A is overwritten with its Cholesky factorization.
        ///
        /// @tparam T Numerical element type (double).
        /// @param a Input symmetric positive definite matrix A (overwritten with Cholesky factorization).
        /// @param b Input-output matrix B (right-hand side, overwritten with solution X).
        /// @throw std::runtime_error if A is not square, B has incorrect size, or DPOTRF/DPOTRS fails.
        template <>
        void SolvePosSysLinEquations<double>(Matrix_<double> &a, Matrix_<double> &b)
        {
            LPCK_INT info = 0;
            LPCK_INT n = a.shape().first;
            LPCK_INT nrhs = b.shape().second;
            CholeskyFactorization(a);
            
#ifdef _APPLE
            dpotrs_(&l_uplo, &n, &nrhs, a.data(), &n, b.data(), &n, &info);
#elif defined(W_MKL)
            info = LAPACKE_dpotrs(l_major, l_uplo, n, nrhs, a.data(), n, b.data(), n);
#else
            LAPACK_dpotrs(&l_uplo, &n, &nrhs, a.data(), &n, b.data(), &n, &info);
#endif
            if (info != 0)
            {
                throw std::runtime_error("Failure in DPOTRS");
            }
        };

        /// @brief Solve a symmetric positive definite system of linear equations A*X = B (single precision)
        ///
        /// Solves the symmetric positive definite linear system A*X = B using Cholesky factorization (SPOTRF)
        /// followed by triangular solve (SPOTRS). Matrix A is assumed to be symmetric and positive definite.
        /// Matrix A is overwritten with its Cholesky factorization.
        ///
        /// @tparam T Numerical element type (float).
        /// @param a Input symmetric positive definite matrix A (overwritten with Cholesky factorization).
        /// @param b Input-output matrix B (right-hand side, overwritten with solution X).
        /// @throw std::runtime_error if A is not square, B has incorrect size, or SPOTRF/SPOTRS fails.
        template <>
        void SolvePosSysLinEquations<float>(Matrix_<float> &a, Matrix_<float> &b)
        {
            LPCK_INT info = 0;
            LPCK_INT n = a.shape().first;
            LPCK_INT nrhs = b.shape().second;
            CholeskyFactorization(a);
            if (static_cast<size_t>(b.shape().first) != static_cast<size_t>(n))
            {
                throw std::runtime_error("B should have as much rows as A. B(n, nrhs)");
            }

#ifdef _APPLE
            spotrs_(&l_uplo, &n, &nrhs, a.data(), &n, b.data(), &n, &info);
#elif defined(W_MKL)
            info = LAPACKE_spotrs(l_major, l_uplo, n, nrhs, a.data(), n, b.data(), n);
#else
            LAPACK_spotrs(&l_uplo, &n, &nrhs, a.data(), &n, b.data(), &n, &info);
#endif
            if (info != 0)
            {
                throw std::runtime_error("Failure in SGETRS");
            }
        };

        /// @brief Solve a general system of linear equations A*X = B or A^T*X = B (double precision)
        ///
        /// Solves the general linear system with optional transpose using LU factorization (DGETRF)
        /// followed by triangular solve (DGETRS). Matrix A is overwritten with its LU factorization.
        ///
        /// @tparam T Numerical element type (double).
        /// @param Ta Transpose option for matrix A: "N" (no transpose, A*X = B), "T" (transpose, A^T*X = B).
        /// @param a Input matrix A (double precision, square, overwritten with LU factorization).
        /// @param b Input-output matrix B (right-hand side, overwritten with solution X).
        /// @throw std::runtime_error if A is not square, B has incorrect size, or factorization/solve fails.
        template <>
        void SolveGenSysLinEquations<double>(const char *Ta, Matrix_<double> &a, Matrix_<double> &b)
        {
            LPCK_INT info = 0;
            LPCK_INT n = a.shape().first;
            LPCK_INT nrhs = b.shape().second;
            Vector<LPCK_INT> ipiv(n);
            LUFactorization(a, ipiv);
            
#ifdef _APPLE
            dgetrs_(Ta, &n, &nrhs, a.data(), &n, ipiv.data(), b.data(), &n, &info);
#elif defined(W_MKL)
            info = LAPACKE_dgetrs(l_major, *Ta, n, nrhs, a.data(), n, ipiv.data(), b.data(), n);
#else
            LAPACK_dgetrs(Ta, &n, &nrhs, a.data(), &n, ipiv.data(), b.data(), &n, &info);
#endif
            if (info != 0)
            {
                throw std::runtime_error("Failure in DGETRS");
            }
        };

        /// @brief Solve a general system of linear equations A*X = B or A^T*X = B (single precision)
        ///
        /// Solves the general linear system with optional transpose using LU factorization (SGETRF)
        /// followed by triangular solve (SGETRS). Matrix A is overwritten with its LU factorization.
        ///
        /// @tparam T Numerical element type (float).
        /// @param Ta Transpose option for matrix A: "N" (no transpose, A*X = B), "T" (transpose, A^T*X = B).
        /// @param a Input matrix A (single precision, square, overwritten with LU factorization).
        /// @param b Input-output matrix B (right-hand side, overwritten with solution X).
        /// @throw std::runtime_error if A is not square, B has incorrect size, or factorization/solve fails.
        template <>
        void SolveGenSysLinEquations<float>(const char *Ta, Matrix_<float> &a, Matrix_<float> &b)
        {
            LPCK_INT info = 0;
            LPCK_INT n = a.shape().first;
            LPCK_INT nrhs = b.shape().second;
            Vector<LPCK_INT> ipiv(n);
            LUFactorization(a, ipiv);
            if (static_cast<size_t>(b.shape().first) != static_cast<size_t>(n))
            {
                throw std::runtime_error("B should have as much rows as A. B(n, nrhs)");
            }

#ifdef _APPLE
            sgetrs_(Ta, &n, &nrhs, a.data(), &n, ipiv.data(), b.data(), &n, &info);
#elif defined(W_MKL)
            info = LAPACKE_sgetrs(l_major, *Ta, n, nrhs, a.data(), n, ipiv.data(), b.data(), n);
#else
            LAPACK_sgetrs(Ta, &n, &nrhs, a.data(), &n, ipiv.data(), b.data(), &n, &info);
#endif
            if (info != 0)
            {
                throw std::runtime_error("Failure in SGETRS");
            }
        };

        /// @brief Solve a general system of linear equations with transpose option (double precision, convenience overload)
        ///
        /// Convenience overload where the transpose option Ta is provided as the last parameter.
        /// Solves A*X = B or A^T*X = B using LU factorization (DGETRF) followed by triangular solve (DGETRS).
        /// Matrix A is overwritten with its LU factorization.
        ///
        /// @tparam T Numerical element type (double).
        /// @param a Input matrix A (double precision, square, overwritten with LU factorization).
        /// @param b Input-output matrix B (right-hand side, overwritten with solution X).
        /// @param Ta Transpose option for matrix A: "N" (no transpose, A*X = B), "T" (transpose, A^T*X = B).
        /// @throw std::runtime_error if A is not square, B has incorrect size, or factorization/solve fails.
        template <>
        void SolveGenSysLinEquations<double>(Matrix_<double> &a, Matrix_<double> &b, const char *Ta)
        {
            LPCK_INT info = 0;
            LPCK_INT n = a.shape().first;
            LPCK_INT nrhs = b.shape().second;
            Vector<LPCK_INT> ipiv(n);
            LUFactorization(a, ipiv);
            if (static_cast<size_t>(b.shape().first) != static_cast<size_t>(n))
            {
                throw std::runtime_error("B should have as much rows as A. B(n, nrhs)");
            }
#ifdef _APPLE
            dgetrs_(Ta, &n, &nrhs, a.data(), &n, ipiv.data(), b.data(), &n, &info);
#elif defined(W_MKL)
            info = LAPACKE_dgetrs(l_major, *Ta, n, nrhs, a.data(), n, ipiv.data(), b.data(), n);
#else
            LAPACK_dgetrs(Ta, &n, &nrhs, a.data(), &n, ipiv.data(), b.data(), &n, &info);
#endif
            if (info != 0)
            {
                throw std::runtime_error("Failure in DGETRS");
            }
        };

        /// @brief Solve a general system of linear equations with transpose option (single precision, convenience overload)
        ///
        /// Convenience overload where the transpose option Ta is provided as the last parameter.
        /// Solves A*X = B or A^T*X = B using LU factorization (SGETRF) followed by triangular solve (SGETRS).
        /// Matrix A is overwritten with its LU factorization.
        ///
        /// @tparam T Numerical element type (float).
        /// @param a Input matrix A (single precision, square, overwritten with LU factorization).
        /// @param b Input-output matrix B (right-hand side, overwritten with solution X).
        /// @param Ta Transpose option for matrix A: "N" (no transpose, A*X = B), "T" (transpose, A^T*X = B).
        /// @throw std::runtime_error if A is not square, B has incorrect size, or factorization/solve fails.
        template <>
        void SolveGenSysLinEquations<float>(Matrix_<float> &a, Matrix_<float> &b, const char *Ta)
        {
            LPCK_INT info = 0;
            LPCK_INT n = a.shape().first;
            LPCK_INT nrhs = b.shape().second;
            Vector<LPCK_INT> ipiv(n);
            LUFactorization(a, ipiv);
            if (static_cast<size_t>(b.shape().first) != static_cast<size_t>(n))
            {
                throw std::runtime_error("B should have as much rows as A. B(n, nrhs)");
            }
#ifdef _APPLE
            sgetrs_(Ta, &n, &nrhs, a.data(), &n, ipiv.data(), b.data(), &n, &info);
#elif defined(W_MKL)
            info = LAPACKE_sgetrs(l_major, *Ta, n, nrhs, a.data(), n, ipiv.data(), b.data(), n);
#else
            LAPACK_sgetrs(Ta, &n, &nrhs, a.data(), &n, ipiv.data(), b.data(), &n, &info);
#endif
            if (info != 0)
            {
                throw std::runtime_error("Failure in SGETRS");
            }
        };

        /// @brief Solve a triangular system T*X = B or T^T*X = B using packed storage (double precision)
        ///
        /// Solves a triangular system where the triangular matrix is stored in packed format (LowTriMatrix_<T>).
        /// Uses DTPTRS for triangular solve. The matrix must be triangular; no factorization is needed.
        ///
        /// @tparam T Numerical element type (double).
        /// @param Ta Transpose option: "N" (no transpose, T*X = B), "T" (transpose, T^T*X = B).
        /// @param a Input triangular matrix A in packed storage format (square, double precision).
        /// @param b Input-output matrix B (right-hand side, overwritten with solution X).
        /// @throw std::runtime_error if A is not square or if DTPTRS fails.
        template <>
        void SolveGenSysLinEquations<double>(const char *Ta, LowTriMatrix_<double> &a, Matrix_<double> &b)
        {
            LPCK_INT info = 0;
            if (a.shape().first != a.shape().second)
            {
                throw std::runtime_error("A should be symmetric");
            };
            LPCK_INT n = a.shape().first;
            LPCK_INT nrhs = b.shape().second;
            if (static_cast<size_t>(b.shape().first) != static_cast<size_t>(n))
            {
                throw std::runtime_error("B should have as much rows as A. B(n, nrhs)");
            }
#ifdef _APPLE
            dtptrs_(&l_uplo, Ta, &l_nondiag, &n, &nrhs, a.data(), b.data(), &n, &info);
#elif defined(W_MKL)
            info = LAPACKE_dtptrs(l_major, l_uplo, *Ta, l_nondiag, n, nrhs, a.data(), b.data(), n);
#else
            LAPACK_dtptrs(&l_uplo, Ta, &l_nondiag, &n, &nrhs, a.data(), b.data(), &n, &info);
#endif
            if (info != 0)
            {
                throw std::runtime_error("Failure in DTPTRS");
            }
        };

        /// @brief Solve a triangular system T*X = B or T^T*X = B using packed storage (single precision)
        ///
        /// Solves a triangular system where the triangular matrix is stored in packed format (LowTriMatrix_<T>).
        /// Uses STPTRS for triangular solve. The matrix must be triangular; no factorization is needed.
        ///
        /// @tparam T Numerical element type (float).
        /// @param Ta Transpose option: "N" (no transpose, T*X = B), "T" (transpose, T^T*X = B).
        /// @param a Input triangular matrix A in packed storage format (square, single precision).
        /// @param b Input-output matrix B (right-hand side, overwritten with solution X).
        /// @throw std::runtime_error if A is not square or if STPTRS fails.
        template <>
        void SolveGenSysLinEquations<float>(const char *Ta, LowTriMatrix_<float> &a, Matrix_<float> &b)
        {
            LPCK_INT info = 0;
            LPCK_INT n = a.shape().first;
            LPCK_INT nrhs = b.shape().second;
            if (static_cast<size_t>(b.shape().first) != static_cast<size_t>(n))
            {
                throw std::runtime_error("B should have as much rows as A. B(n, nrhs)");
            }
#ifdef _APPLE
            stptrs_(&l_uplo, Ta, &l_nondiag, &n, &nrhs, a.data(), b.data(), &n, &info);
#elif defined(W_MKL)
            info = LAPACKE_stptrs(l_major, l_uplo, *Ta, l_nondiag, n, nrhs, a.data(), b.data(), n);
#else
            LAPACK_stptrs(&l_uplo, Ta, &l_nondiag, &n, &nrhs, a.data(), b.data(), &n, &info);
#endif
            if (info != 0)
            {
                throw std::runtime_error("Failure in STPTRS");
            }
        };

        /// @brief Solve a triangular system with transpose option (double precision, convenience overload)
        ///
        /// Convenience overload where the transpose option Ta is provided as the last parameter.
        /// Solves T*X = B or T^T*X = B using packed triangular storage format via DTPTRS.
        ///
        /// @tparam T Numerical element type (double).
        /// @param a Input triangular matrix A in packed storage format (square, double precision).
        /// @param b Input-output matrix B (right-hand side, overwritten with solution X).
        /// @param Ta Transpose option: "N" (no transpose, T*X = B), "T" (transpose, T^T*X = B).
        /// @throw std::runtime_error if A is not square or if DTPTRS fails.
        template <>
        void SolveGenSysLinEquations<double>(LowTriMatrix_<double> &a, Matrix_<double> &b, const char *Ta)
        {
            LPCK_INT info = 0;
            LPCK_INT n = a.shape().first;
            LPCK_INT nrhs = b.shape().second;
            if (static_cast<size_t>(b.shape().first) != static_cast<size_t>(n))
            {
                throw std::runtime_error("B should have as much rows as A. B(n, nrhs)");
            }
#ifdef _APPLE
            dtptrs_(&l_uplo, Ta, &l_nondiag, &n, &nrhs, a.data(), b.data(), &n, &info);
#elif defined(W_MKL)
            info = LAPACKE_dtptrs(l_major, l_uplo, *Ta, l_nondiag, n, nrhs, a.data(), b.data(), n);
#else
            LAPACK_dtptrs(&l_uplo, Ta, &l_nondiag, &n, &nrhs, a.data(), b.data(), &n, &info);
#endif
            if (info != 0)
            {
                throw std::runtime_error("Failure in DTPTRS");
            }
        };

        /// @brief Solve a triangular system with transpose option (single precision, convenience overload)
        ///
        /// Convenience overload where the transpose option Ta is provided as the last parameter.
        /// Solves T*X = B or T^T*X = B using packed triangular storage format via STPTRS.
        ///
        /// @tparam T Numerical element type (float).
        /// @param a Input triangular matrix A in packed storage format (square, single precision).
        /// @param b Input-output matrix B (right-hand side, overwritten with solution X).
        /// @param Ta Transpose option: "N" (no transpose, T*X = B), "T" (transpose, T^T*X = B).
        /// @throw std::runtime_error if A is not square or if STPTRS fails.
        template <>
        void SolveGenSysLinEquations<float>(LowTriMatrix_<float> &a, Matrix_<float> &b, const char *Ta)
        {
            LPCK_INT info = 0;
            LPCK_INT n = a.shape().first;
            LPCK_INT nrhs = b.shape().second;
            if (static_cast<size_t>(b.shape().first) != static_cast<size_t>(n))
            {
                throw std::runtime_error("B should have as much rows as A. B(n, nrhs)");
            }
#ifdef _APPLE
            stptrs_(&l_uplo, Ta, &l_nondiag, &n, &nrhs, a.data(), b.data(), &n, &info);
#elif defined(W_MKL)
            info = LAPACKE_stptrs(l_major, l_uplo, *Ta, l_nondiag, n, nrhs, a.data(), b.data(), n);
#else
            LAPACK_stptrs(&l_uplo, Ta, &l_nondiag, &n, &nrhs, a.data(), b.data(), &n, &info);
#endif
            if (info != 0)
            {
                throw std::runtime_error("Failure in STPTRS");
            }
        };

        /// @brief Solve a symmetric indefinite system A*X = B (double precision)
        ///
        /// Solves a symmetric indefinite system stored in packed format using DSPTRF (symmetric indefinite
        /// factorization) followed by DSPTRS (symmetric indefinite solve). The factorization uses Bunch-Kaufman
        /// pivoting, which is more numerically stable than Cholesky for indefinite matrices.
        ///
        /// @tparam T Numerical element type (double).
        /// @param a Input symmetric indefinite matrix A in packed storage format (square, double precision).
        /// @param b Input-output matrix B (right-hand side, overwritten with solution X).
        /// @throw std::runtime_error if A is not square, B has incorrect size, or factorization/solve fails.
        template <>
        void SolveSymSysLinEquations<double>(LowTriMatrix_<double> &a, Matrix_<double> &b)
        {
            LPCK_INT info = 0;
            LPCK_INT n = a.shape().first;
            LPCK_INT nrhs = b.shape().second;
            if (static_cast<size_t>(b.shape().first) != static_cast<size_t>(n))
            {
                throw std::runtime_error("B should have as much rows as A. B(n, nrhs)");
            }
            Vector<LPCK_INT> ipiv(n);
#ifdef _APPLE
            dsptrf_(&l_uplo, &n, a.data(), ipiv.data(), &info);
#elif defined(W_MKL)
            info = LAPACKE_dsptrf(l_major, l_uplo, n, a.data(), ipiv.data());
#else
            LAPACK_dsptrf(&l_uplo, &n, a.data(), ipiv.data(), &info);
#endif
            if (info != 0)
            {
                throw std::runtime_error("Failure in DSPTRF");
            }
#ifdef _APPLE
            dsptrs_(&l_uplo, &n, &nrhs, a.data(), ipiv.data(), b.data(), &n, &info);
#elif defined(W_MKL)
            info = LAPACKE_dsptrs(l_major, l_uplo, n, nrhs, a.data(), ipiv.data(), b.data(), n);
#else
            LAPACK_dsptrs(&l_uplo, &n, &nrhs, a.data(), ipiv.data(), b.data(), &n, &info);
#endif
            if (info != 0)
            {
                throw std::runtime_error("Failure in DSPTRS");
            }
        };

        /// @brief Solve a symmetric indefinite system A*X = B (single precision)
        ///
        /// Solves a symmetric indefinite system stored in packed format using SSPTRF (symmetric indefinite
        /// factorization) followed by SSPTRS (symmetric indefinite solve). The factorization uses Bunch-Kaufman
        /// pivoting, which is more numerically stable than Cholesky for indefinite matrices.
        ///
        /// @tparam T Numerical element type (float).
        /// @param a Input symmetric indefinite matrix A in packed storage format (square, single precision).
        /// @param b Input-output matrix B (right-hand side, overwritten with solution X).
        /// @throw std::runtime_error if A is not square, B has incorrect size, or factorization/solve fails.
        template <>
        void SolveSymSysLinEquations<float>(LowTriMatrix_<float> &a, Matrix_<float> &b)
        {
            LPCK_INT info = 0;
            LPCK_INT n = a.shape().first;
            LPCK_INT nrhs = b.shape().second;
            if (static_cast<size_t>(b.shape().first) != static_cast<size_t>(n))
            {
                throw std::runtime_error("B should have as much rows as A. B(n, nrhs)");
            }
            Vector<LPCK_INT> ipiv(n);
#ifdef _APPLE
            ssptrf_(&l_uplo, &n, a.data(), ipiv.data(), &info);
#elif defined(W_MKL)
            info = LAPACKE_ssptrf(l_major, l_uplo, n, a.data(), ipiv.data());
#else
            LAPACK_ssptrf(&l_uplo, &n, a.data(), ipiv.data(), &info);
#endif
            if (info != 0)
            {
                throw std::runtime_error("Failure in SSPTRF");
            }
#ifdef _APPLE
            ssptrs_(&l_uplo, &n, &nrhs, a.data(), ipiv.data(), b.data(), &n, &info);
#elif defined(W_MKL)
            info = LAPACKE_ssptrs(l_major, l_uplo, n, nrhs, a.data(), ipiv.data(), b.data(), n);
#else
            LAPACK_ssptrs(&l_uplo, &n, &nrhs, a.data(), ipiv.data(), b.data(), &n, &info);
#endif
            if (info != 0)
            {
                throw std::runtime_error("Failure in SSPTRS");
            }
        };

        /// @brief Compute the inverse of a double-precision triangular matrix
        ///
        /// Computes the inverse of a triangular matrix (DTRTRI). The matrix is overwritten with its inverse.
        /// Only the lower or upper triangular part is used (determined by l_uplo defined in const.h).
        ///
        /// @tparam T Numerical element type (double).
        /// @param a Input triangular matrix (double precision, square, overwritten with its inverse).
        /// @throw std::runtime_error if A is not square or if DTRTRI fails (singular matrix).
        template <>
        void InvertTriMatrix<double>(Matrix_<double> &a)
        {
            LPCK_INT info = 0;
            LPCK_INT n = a.shape().first;
#ifdef _APPLE
            dtrtri_(&l_uplo, &l_nondiag, &n, a.data(), &n, &info);
#elif defined(W_MKL)
            info = LAPACKE_dtrtri(l_major, l_uplo, l_nondiag, n, a.data(), n);
#else
            LAPACK_dtrtri(&l_uplo, &l_nondiag, &n, a.data(), &n, &info);
#endif
            if (info != 0)
            {
                throw std::runtime_error("Failure in DTRTRI");
            }
        };

        /// @brief Compute the inverse of a single-precision triangular matrix
        ///
        /// Computes the inverse of a triangular matrix (STRTRI). The matrix is overwritten with its inverse.
        /// Only the lower or upper triangular part is used (determined by l_uplo defined in const.h).
        ///
        /// @tparam T Numerical element type (float).
        /// @param a Input triangular matrix (single precision, square, overwritten with its inverse).
        /// @throw std::runtime_error if A is not square or if STRTRI fails (singular matrix).
        template <>
        void InvertTriMatrix<float>(Matrix_<float> &a)
        {
            LPCK_INT info = 0;
            LPCK_INT n = a.shape().first;

#ifdef _APPLE
            strtri_(&l_uplo, &l_nondiag, &n, a.data(), &n, &info);
#elif defined(W_MKL)
            info = LAPACKE_strtri(l_major, l_uplo, l_nondiag, n, a.data(), n);
#else
            LAPACK_strtri(&l_uplo, &l_nondiag, &n, a.data(), &n, &info);
#endif
            if (info != 0)
            {
                throw std::runtime_error("Failure in STRTRI");
            }
        };

/// @brief Compute eigenvalues and optionally eigenvectors of a double-precision symmetric matrix
        ///
        /// Computes the eigenvalues and optionally eigenvectors of a symmetric matrix using the
        /// divide-and-conquer method (DSYEVD). If l_jobz='N', only eigenvalues are computed.
        /// If l_jobz='V', both eigenvalues and eigenvectors are computed. Matrix A is overwritten
        /// with the eigenvectors (if computed).
        ///
        /// @tparam T Numerical element type (double).
        /// @param a Input symmetric matrix (double precision, square, overwritten with eigenvectors if l_jobz='V').
        /// @param eigenvalues Output vector of eigenvalues (sorted in ascending order).
        /// @param l_jobz Compute eigenvalues only (l_jobz='N') or eigenvalues and eigenvectors (l_jobz='V'). Default: 'V'.
        /// @throw std::runtime_error if A is not square, invalid l_jobz, or DSYEVD fails.
        template <>
        void SymEigenvalueDecomposition<double>(Matrix_<double> &a, Vector_<double>& eigenvalues, char l_jobz)
        {    
            LPCK_INT info = 0;      
            LPCK_INT n = a.shape().first;
#ifndef W_MKL   
            if (a.shape().first != a.shape().second)
            {
                throw std::runtime_error("A should be symmetric");
            };
            
            if (l_jobz != 'N' && l_jobz != 'V')
            {
                throw std::runtime_error("l_jobz should be 'N' or 'V'");
            }
            LPCK_INT size_work = 2 * n - 1;
            LPCK_INT size_iwork = 1;
            if (l_jobz == 'V')
            {
                size_work = 1 + 6 * n + 2 * n * n;
                size_iwork = 3 + 5 * n;
            }
            
            Vector<double> work(size_work);
            Vector<LPCK_INT> iwork(size_iwork);
#endif
#ifdef _APPLE
            dsyevd_(&l_jobz, &l_uplo, &n, a.data(), &n, eigenvalues.data(), work.data(), &size_work, iwork.data(),
            &size_iwork, &info);
#elif defined(W_MKL)
            info = LAPACKE_dsyevd(l_major, l_jobz, l_uplo, n, a.data(), n, eigenvalues.data());
#else
            LAPACK_dsyevd(&l_jobz, &l_uplo, &n, a.data(), &n, eigenvalues.data(), work.data(), &size_work, iwork.data(),
            &size_iwork, &info);
#endif
            if (info != 0)
            {
                throw std::runtime_error("Failure in DSYEV");
            }
        };

        /// @brief Compute eigenvalues and optionally eigenvectors of a single-precision symmetric matrix
        ///
        /// Computes the eigenvalues and optionally eigenvectors of a symmetric matrix using the
        /// divide-and-conquer method (SSYEVD). If l_jobz='N', only eigenvalues are computed.
        /// If l_jobz='V', both eigenvalues and eigenvectors are computed. Matrix A is overwritten
        /// with the eigenvectors (if computed).
        ///
        /// @tparam T Numerical element type (float).
        /// @param a Input symmetric matrix (single precision, square, overwritten with eigenvectors if l_jobz='V').
        /// @param eigenvalues Output vector of eigenvalues (sorted in ascending order).
        /// @param l_jobz Compute eigenvalues only (l_jobz='N') or eigenvalues and eigenvectors (l_jobz='V'). Default: 'V'.
        /// @throw std::runtime_error if A is not square, invalid l_jobz, or SSYEVD fails.
        template <>
        void SymEigenvalueDecomposition<float>(Matrix_<float> &a, Vector_<float>& eigenvalues, char l_jobz)
        {
            
            LPCK_INT info = 0;
            LPCK_INT n = a.shape().first;
#ifndef W_MKL
            if (a.shape().first != a.shape().second)
            {
                throw std::runtime_error("A should be symmetric");
            };
            
            if (l_jobz != 'N' && l_jobz != 'V')
            {
                throw std::runtime_error("l_jobz should be 'N' or 'V'");
            }
            LPCK_INT size_work = 2 * n - 1;
            LPCK_INT size_iwork = 1;
            if (l_jobz == 'V')
            {
                size_work = 1 + 6 * n + 2 * n * n;
                size_iwork = 3 + 5 * n;
            }
            
            Vector<float> work(size_work);
            Vector<LPCK_INT> iwork(size_iwork);
#endif
#ifdef _APPLE
            ssyevd_(&l_jobz, &l_uplo, &n, a.data(), &n, eigenvalues.data(), work.data(), &size_work, iwork.data(),
            &size_iwork, &info);
#elif defined(W_MKL)
            info = LAPACKE_ssyevd(l_major, l_jobz, l_uplo, n, a.data(), n, eigenvalues.data());
#else
            LAPACK_ssyevd(&l_jobz, &l_uplo, &n, a.data(), &n, eigenvalues.data(), work.data(), &size_work, iwork.data(),
            &size_iwork, &info);
#endif
            if (info != 0)
            {
                throw std::runtime_error("Failure in DSYEV");
            }
        }; 
        
    } // namespace cpu
} // namespace lahva
