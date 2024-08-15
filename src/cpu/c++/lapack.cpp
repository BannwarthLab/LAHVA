#include "impl/cpu/lapack.hpp"
#include <stdexcept>
namespace tcgmtensor{

    namespace cpu
    {
        template<>
        void SolveGenSysLinEquations<double>(const char* Ta, Matrix<double>& a, Matrix<double>& b)
        {
            LPCK_INT info = 0;
            if ( a.shape().first != a.shape().second)
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

        template<>
        void SolveGenSysLinEquations<float>(const char* Ta, Matrix<float>& a, Matrix<float>& b)
        {
            LPCK_INT info = 0;
            if ( a.shape().first != a.shape().second)
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

        template<>
        void SolveGenSysLinEquations<double>(Matrix<double>& a, Matrix<double>& b, const char* Ta)
        {
            LPCK_INT info = 0;
            if ( a.shape().first != a.shape().second)
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

        template<>
        void SolveGenSysLinEquations<float>(Matrix<float>& a, Matrix<float>& b, const char* Ta)
        {
            LPCK_INT info = 0;
            if ( a.shape().first != a.shape().second)
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

        template void SolveGenSysLinEquations<double>(const char* Ta, Matrix<double>& a, Matrix<double>& b);
        template void SolveGenSysLinEquations<float>(const char* Ta, Matrix<float>& a, Matrix<float>& b);
        template void SolveGenSysLinEquations<double>(Matrix<double>& a, Matrix<double>& b, const char* Ta);
        template void SolveGenSysLinEquations<float>(Matrix<float>& a, Matrix<float>& b, const char* Ta);

        
    } // namespace cpu
    
}
