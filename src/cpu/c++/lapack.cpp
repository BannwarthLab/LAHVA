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
            assert(ipiv.size() == n);
            dgetrf_(&n, &n, a.data(), &n, ipiv.data(), &info);
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
            assert(ipiv.size() == n);
            sgetrf_(&n, &n, a.data(), &n, ipiv.data(), &info);
            if (info != 0)
            {
                throw std::runtime_error("Failure in DGETRF");
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
            
            dgetrs_(Ta, &n, &nrhs, a.data(), &n, ipiv.data(), b.data(), &n, &info);
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

            sgetrs_(Ta, &n, &nrhs, a.data(), &n, ipiv.data(), b.data(), &n, &info);
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
            dgetrs_(Ta, &n, &nrhs, a.data(), &n, ipiv.data(), b.data(), &n, &info);
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
            sgetrs_(Ta, &n, &nrhs, a.data(), &n, ipiv.data(), b.data(), &n, &info);
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
            dtptrs_(&l_uplo, Ta, &l_nondiag, &n, &nrhs, a.data(), b.data(), &n, &info);
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
            stptrs_(&l_uplo, Ta, &l_nondiag, &n, &nrhs, a.data(), b.data(), &n, &info);
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
            dtptrs_(&l_uplo, Ta, &l_nondiag, &n, &nrhs, a.data(), b.data(), &n, &info);
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
            stptrs_(&l_uplo, Ta, &l_nondiag, &n, &nrhs, a.data(), b.data(), &n, &info);
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
            dsptrf_(&l_uplo, &n, a.data(), ipiv.data(), &info);
            if (info != 0)
            {
                throw std::runtime_error("Failure in DSPTRF");
            }
            dsptrs_(&l_uplo, &n, &nrhs, a.data(), ipiv.data(), b.data(), &n, &info);
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
            ssptrf_(&l_uplo, &n, a.data(), ipiv.data(), &info);
            if (info != 0)
            {
                throw std::runtime_error("Failure in SSPTRF");
            }
            ssptrs_(&l_uplo, &n, &nrhs, a.data(), ipiv.data(), b.data(), &n, &info);
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
            
            dtrtri_(&l_uplo, &l_nondiag, &n, a.data(), &n, &info);
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
            
            strtri_(&l_uplo, &l_nondiag, &n, a.data(), &n, &info);
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
            Vector<double> w(n);
            
            dsyevd_(&l_jobz, &l_uplo, &n, a.data(), &n, eigenvalues.data(), work.data(), &size_work, iwork.data(),
            &size_iwork, &info);
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
            Vector<float> w(n);
            
            ssyevd_(&l_jobz, &l_uplo, &n, a.data(), &n, eigenvalues.data(), work.data(), &size_work, iwork.data(),
            &size_iwork, &info);
            if (info != 0)
            {
                throw std::runtime_error("Failure in DSYEV");
            }
        }; 
        
    } // namespace cpu

}
