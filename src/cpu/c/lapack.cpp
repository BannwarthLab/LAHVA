#include "impl/cpu/lapack.h"
#include <stdexcept>
namespace tcgmtensor{

    namespace cpu
    {
        void SolveGenSysLinEquations(const char* T, const LPCK_INT n, double* a, const LPCK_INT nrhs, double* b)
        {
            LPCK_INT info = 0;
            LPCK_INT* ipiv = new LPCK_INT[n];

            info = LAPACKE_dgetrf(l_major, n, n, a, n, ipiv);
            if (info != 0)
            {
                throw std::runtime_error("Failure in DGETRF");
            }
            info = LAPACKE_dgetrs(l_major, *T, n, nrhs, a, n, ipiv, b, n);
            if (info != 0)
            {
                throw std::runtime_error("Failure in DGETRS");
            }
            delete[] ipiv;
        };

        void SolveGenSysLinEquations(const LPCK_INT n, double* a, const LPCK_INT nrhs, double* b, const char* T)
        {
            LPCK_INT info = 0;
            LPCK_INT* ipiv = new LPCK_INT[n];

            info = LAPACKE_dgetrf(l_major, n, n, a, n, ipiv);
            if (info != 0)
            {
                throw std::runtime_error("Failure in DGETRF");
            }
            info = LAPACKE_dgetrs(l_major, *T, n, nrhs, a, n, ipiv, b, n);
            if (info != 0)
            {
                throw std::runtime_error("Failure in DGETRS");
            }
            delete[] ipiv;
        };


        void SolveGenSysLinEquations(const char* T, const LPCK_INT n, float* a, const LPCK_INT nrhs, float* b)
        {
            LPCK_INT info = 0;
            LPCK_INT* ipiv = new LPCK_INT[n];

            info = LAPACKE_sgetrf(l_major, n, n, a, n, ipiv);
            if (info != 0)
            {
                throw std::runtime_error("Failure in SGETRF");
            }
            info = LAPACKE_sgetrs(l_major, *T, n, nrhs, a, n, ipiv, b, n);
            if (info != 0)
            {
                throw std::runtime_error("Failure in SGETRS");
            }
            delete[] ipiv;
        };

        void SolveGenSysLinEquations(const LPCK_INT n, float* a, const LPCK_INT nrhs, float* b, const char* T)
        {
            LPCK_INT info = 0;
            LPCK_INT* ipiv = new LPCK_INT[n];

            info = LAPACKE_sgetrf(l_major, n, n, a, n, ipiv);
            if (info != 0)
            {
                throw std::runtime_error("Failure in SGETRF");
            }
            info = LAPACKE_sgetrs(l_major, *T, n, nrhs, a, n, ipiv, b, n);
            if (info != 0)
            {
                throw std::runtime_error("Failure in SGETRS");
            }
            delete[] ipiv;
        };
    } // namespace cpu
    
}
