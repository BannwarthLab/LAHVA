#include "impl/cpu/lapack.hpp"
#include <stdexcept>
namespace tcgmtensor
{

    namespace cpu
    {
        template <>
        void SolveGenSysLinEquations<double>(const char *Ta, Matrix<double> &a, Matrix<double> &b)
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
            info = LAPACKE_dgetrf(l_major, n, n, a.data(), n, ipiv.data());
            if (info != 0)
            {
                throw std::runtime_error("Failure in DGETRF");
            }
            info = LAPACKE_dgetrs(l_major, *Ta, n, nrhs, a.data(), n, ipiv.data(), b.data(), n);
            if (info != 0)
            {
                throw std::runtime_error("Failure in DGETRS");
            }
        };

        template <>
        void SolveGenSysLinEquations<float>(const char *Ta, Matrix<float> &a, Matrix<float> &b)
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
            info = LAPACKE_sgetrf(l_major, n, n, a.data(), n, ipiv.data());
            if (info != 0)
            {
                throw std::runtime_error("Failure in SGETRF");
            }
            info = LAPACKE_sgetrs(l_major, *Ta, n, nrhs, a.data(), n, ipiv.data(), b.data(), n);
            if (info != 0)
            {
                throw std::runtime_error("Failure in SGETRS");
            }
        };

        template <>
        void SolveGenSysLinEquations<double>(Matrix<double> &a, Matrix<double> &b, const char *Ta)
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
            info = LAPACKE_dgetrf(l_major, n, n, a.data(), n, ipiv.data());
            if (info != 0)
            {
                throw std::runtime_error("Failure in DGETRF");
            }
            info = LAPACKE_dgetrs(l_major, *Ta, n, nrhs, a.data(), n, ipiv.data(), b.data(), n);
            if (info != 0)
            {
                throw std::runtime_error("Failure in DGETRS");
            }
        };

        template <>
        void SolveGenSysLinEquations<float>(Matrix<float> &a, Matrix<float> &b, const char *Ta)
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
            info = LAPACKE_sgetrf(l_major, n, n, a.data(), n, ipiv.data());
            if (info != 0)
            {
                throw std::runtime_error("Failure in SGETRF");
            }
            info = LAPACKE_sgetrs(l_major, *Ta, n, nrhs, a.data(), n, ipiv.data(), b.data(), n);
            if (info != 0)
            {
                throw std::runtime_error("Failure in SGETRS");
            }
        };

        template <>
        void SolveGenSysLinEquations<double>(const char *Ta, LowTriMatrix<double> &a, Matrix<double> &b)
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
           // Vector<LPCK_INT> ipiv(n);
            //info = LAPACKE_dsptrf(l_major, l_uplo, n, a.data(), ipiv.data());
            //if (info != 0)
            //{
                //throw std::runtime_error("Failure in DSPTRF");
            //}
            info = LAPACKE_dtptrs(l_major, l_uplo, *Ta, 'N', n, nrhs, a.data(), b.data(), n);
            if (info != 0)
            {
                throw std::runtime_error("Failure in DTPTRS");
            }
        };

        template <>
        void SolveGenSysLinEquations<float>(const char *Ta, LowTriMatrix<float> &a, Matrix<float> &b)
        {
            LPCK_INT info = 0;
            LPCK_INT n = a.shape().first;
            LPCK_INT nrhs = b.shape().second;
            if (b.shape().first != n)
            {
                throw std::runtime_error("B should have as much rows as A. B(n, nrhs)");
            }
            //Vector<LPCK_INT> ipiv(n);
            //info = LAPACKE_ssptrf(l_major, l_uplo, n, a.data(), ipiv.data());
            //if (info != 0)
            //{
                //throw std::runtime_error("Failure in SSPTRF");
            //}
            info = LAPACKE_stptrs(l_major, l_uplo, *Ta, 'N', n, nrhs, a.data(), b.data(), n);
            if (info != 0)
            {
                throw std::runtime_error("Failure in STPTRS");
            }
        };

        template <>
        void SolveGenSysLinEquations<double>(LowTriMatrix<double> &a, Matrix<double> &b, const char *Ta)
        {
            LPCK_INT info = 0;
            LPCK_INT n = a.shape().first;
            LPCK_INT nrhs = b.shape().second;
            if (b.shape().first != n)
            {
                throw std::runtime_error("B should have as much rows as A. B(n, nrhs)");
            }
            //Vector<LPCK_INT> ipiv(n);
            //info = LAPACKE_dsptrf(l_major, l_uplo, n, a.data(), ipiv.data());
            //if (info != 0)
            //{
                //throw std::runtime_error("Failure in DSPTRF");
            //}
            info = LAPACKE_dtptrs(l_major, l_uplo, *Ta, 'N', n, nrhs, a.data(), b.data(), n);
            if (info != 0)
            {
                throw std::runtime_error("Failure in DTPTRS");
            }
        };

        template <>
        void SolveGenSysLinEquations<float>(LowTriMatrix<float> &a, Matrix<float> &b, const char *Ta)
        {
            LPCK_INT info = 0;
            LPCK_INT n = a.shape().first;
            LPCK_INT nrhs = b.shape().second;
            if (b.shape().first != n)
            {
                throw std::runtime_error("B should have as much rows as A. B(n, nrhs)");
            }
            //Vector<LPCK_INT> ipiv(n);
            //info = LAPACKE_ssptrf(l_major, l_uplo, n, a.data(), ipiv.data());
            //if (info != 0)
            //{
                //throw std::runtime_error("Failure in SSPTRF");
            //}
            info = LAPACKE_stptrs(l_major, l_uplo, *Ta, 'N', n, nrhs, a.data(), b.data(), n);
            if (info != 0)
            {
                throw std::runtime_error("Failure in STPTRS");
            }
        };

        template <>
        void SolveSymSysLinEquations<double>(LowTriMatrix<double> &a, Matrix<double> &b)
        {
            LPCK_INT info = 0;
            LPCK_INT n = a.shape().first;
            LPCK_INT nrhs = b.shape().second;
            if (b.shape().first != n)
            {
                throw std::runtime_error("B should have as much rows as A. B(n, nrhs)");
            }
            Vector<LPCK_INT> ipiv(n);
            info = LAPACKE_dsptrf(l_major, l_uplo, n, a.data(), ipiv.data());
            if (info != 0)
            {
                throw std::runtime_error("Failure in DSPTRF");
            }
            info = LAPACKE_dsptrs(l_major, l_uplo, n, nrhs, a.data(), ipiv.data(), b.data(), n);
            if (info != 0)
            {
                throw std::runtime_error("Failure in DSPTRS");
            }
        };

        template <>
        void SolveSymSysLinEquations<float>(LowTriMatrix<float> &a, Matrix<float> &b)
        {
            LPCK_INT info = 0;
            LPCK_INT n = a.shape().first;
            LPCK_INT nrhs = b.shape().second;
            if (b.shape().first != n)
            {
                throw std::runtime_error("B should have as much rows as A. B(n, nrhs)");
            }
            Vector<LPCK_INT> ipiv(n);
            info = LAPACKE_ssptrf(l_major, l_uplo, n, a.data(), ipiv.data());
            if (info != 0)
            {
                throw std::runtime_error("Failure in SSPTRF");
            }
            info = LAPACKE_ssptrs(l_major, l_uplo, n, nrhs, a.data(), ipiv.data(), b.data(), n);
            if (info != 0)
            {
                throw std::runtime_error("Failure in SSPTRS");
            }
        };

        template void SolveGenSysLinEquations<double>(const char *Ta, Matrix<double> &a, Matrix<double> &b);
        template void SolveGenSysLinEquations<float>(const char *Ta, Matrix<float> &a, Matrix<float> &b);
        template void SolveGenSysLinEquations<double>(Matrix<double> &a, Matrix<double> &b, const char *Ta);
        template void SolveGenSysLinEquations<float>(Matrix<float> &a, Matrix<float> &b, const char *Ta);
        template void SolveGenSysLinEquations<double>(const char *Ta, LowTriMatrix<double> &a, Matrix<double> &b);
        template void SolveGenSysLinEquations<float>(const char *Ta, LowTriMatrix<float> &a, Matrix<float> &b);
        template void SolveGenSysLinEquations<double>(LowTriMatrix<double> &a, Matrix<double> &b, const char *Ta);
        template void SolveGenSysLinEquations<float>(LowTriMatrix<float> &a, Matrix<float> &b, const char *Ta);
        template void SolveSymSysLinEquations<double>(LowTriMatrix<double> &a, Matrix<double> &b);
        template void SolveSymSysLinEquations<float>(LowTriMatrix<float> &a, Matrix<float> &b);
        
    } // namespace cpu

}
