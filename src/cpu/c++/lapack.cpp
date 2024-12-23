#include "impl/blas/cpu/lapack.hpp"
#include <stdexcept>
namespace tcgmtensor
{

    namespace cpu
    {
        template <>
        void SolveGenSysLinEquations<double>(const char *Ta, Matrix_<double> &a, Matrix_<double> &b)
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
            Vector<LPCK_INT> ipiv(n);
            dgetrf_(&n, &n, a.data(), &n, ipiv.data(), &info);
            if (info != 0)
            {
                throw std::runtime_error("Failure in DGETRF");
            }
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
            Vector<LPCK_INT> ipiv(n);
            sgetrf_(&n, &n, a.data(), &n, ipiv.data(), &info);
            if (info != 0)
            {
                throw std::runtime_error("Failure in SGETRF");
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
            Vector<LPCK_INT> ipiv(n);
            dgetrf_(&n, &n, a.data(), &n, ipiv.data(), &info);
            if (info != 0)
            {
                throw std::runtime_error("Failure in DGETRF");
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
            Vector<LPCK_INT> ipiv(n);
            sgetrf_(&n, &n, a.data(), &n, ipiv.data(), &info);
            if (info != 0)
            {
                throw std::runtime_error("Failure in SGETRF");
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

        
    } // namespace cpu

}
