#include "impl/blas/cpu/lapack.h"
#include <stdexcept>
namespace lahva{

    namespace cpu
    {
        void SolveGenSysLinEquations(const char* T, const LPCK_INT n, double* a, const LPCK_INT nrhs, double* b)
        {
            LPCK_INT info = 0;
            LPCK_INT* ipiv = new LPCK_INT[n];

            dgetrf_(&n, &n, a, &n, ipiv, &info);
            if (info != 0)
            {
                throw std::runtime_error("Failure in DGETRF");
            }
            dgetrs_(T, &n, &nrhs, a, &n, ipiv, b, &n, &info);
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

            dgetrf_(&n, &n, a, &n, ipiv, &info);
            if (info != 0)
            {
                throw std::runtime_error("Failure in DGETRF");
            }
            dgetrs_(T, &n, &nrhs, a, &n, ipiv, b, &n, &info);
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

            sgetrf_(&n, &n, a, &n, ipiv, &info);
            if (info != 0)
            {
                throw std::runtime_error("Failure in SGETRF");
            }
            sgetrs_(T, &n, &nrhs, a, &n, ipiv, b, &n, &info);
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

            sgetrf_(&n, &n, a, &n, ipiv, &info);
            if (info != 0)
            {
                throw std::runtime_error("Failure in SGETRF");
            }
            sgetrs_(T, &n, &nrhs, a, &n, ipiv, b, &n, &info);
            if (info != 0)
            {
                throw std::runtime_error("Failure in SGETRS");
            }
            delete[] ipiv;
        };
    } // namespace cpu
    
}
