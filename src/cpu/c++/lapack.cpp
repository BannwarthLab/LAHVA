#include "linalg.hpp"
#include "impl/blas/cpu/lapack.hpp"
#include <stdexcept>
#include "lapack_wrap.hpp"
namespace lahva
{

    namespace cpu
    {

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

        void LUFactorization(Matrix_<float>& a, Vector<LPCK_INT>& ipiv)
        {
            LPCK_INT info = 0;
            if (a.shape().first != a.shape().second)
            {
                throw std::runtime_error("A should be symmetric");
            };
            
            LPCK_INT n = a.shape().first;
            if (ipiv.size() != n)
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

        template <>
        void SolvePosSysLinEquations<float>(Matrix_<float> &a, Matrix_<float> &b)
        {
            LPCK_INT info = 0;
            LPCK_INT n = a.shape().first;
            LPCK_INT nrhs = b.shape().second;
            CholeskyFactorization(a);
            if (b.shape().first != n)
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

        template <>
        void SolveGenSysLinEquations<float>(const char *Ta, Matrix_<float> &a, Matrix_<float> &b)
        {
            LPCK_INT info = 0;
            LPCK_INT n = a.shape().first;
            LPCK_INT nrhs = b.shape().second;
            Vector<LPCK_INT> ipiv(n);
            LUFactorization(a, ipiv);
            if (b.shape().first != n)
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

        template <>
        void SolveGenSysLinEquations<double>(Matrix_<double> &a, Matrix_<double> &b, const char *Ta)
        {
            LPCK_INT info = 0;
            LPCK_INT n = a.shape().first;
            LPCK_INT nrhs = b.shape().second;
            Vector<LPCK_INT> ipiv(n);
            LUFactorization(a, ipiv);
            if (b.shape().first != n)
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

        template <>
        void SolveGenSysLinEquations<float>(Matrix_<float> &a, Matrix_<float> &b, const char *Ta)
        {
            LPCK_INT info = 0;
            LPCK_INT n = a.shape().first;
            LPCK_INT nrhs = b.shape().second;
            Vector<LPCK_INT> ipiv(n);
            LUFactorization(a, ipiv);
            if (b.shape().first != n)
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
            if (b.shape().first != n)
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

        template <>
        void SolveGenSysLinEquations<float>(const char *Ta, LowTriMatrix_<float> &a, Matrix_<float> &b)
        {
            LPCK_INT info = 0;
            LPCK_INT n = a.shape().first;
            LPCK_INT nrhs = b.shape().second;
            if (b.shape().first != n)
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

        template <>
        void SolveGenSysLinEquations<double>(LowTriMatrix_<double> &a, Matrix_<double> &b, const char *Ta)
        {
            LPCK_INT info = 0;
            LPCK_INT n = a.shape().first;
            LPCK_INT nrhs = b.shape().second;
            if (b.shape().first != n)
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

        template <>
        void SolveGenSysLinEquations<float>(LowTriMatrix_<float> &a, Matrix_<float> &b, const char *Ta)
        {
            LPCK_INT info = 0;
            LPCK_INT n = a.shape().first;
            LPCK_INT nrhs = b.shape().second;
            if (b.shape().first != n)
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

        template <>
        void SolveSymSysLinEquations<double>(LowTriMatrix_<double> &a, Matrix_<double> &b)
        {
            LPCK_INT info = 0;
            LPCK_INT n = a.shape().first;
            LPCK_INT nrhs = b.shape().second;
            if (b.shape().first != n)
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

        template <>
        void SolveSymSysLinEquations<float>(LowTriMatrix_<float> &a, Matrix_<float> &b)
        {
            LPCK_INT info = 0;
            LPCK_INT n = a.shape().first;
            LPCK_INT nrhs = b.shape().second;
            if (b.shape().first != n)
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
                throw std::runtime_error("Failure in DTRTRI");
            }
        };

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

}
