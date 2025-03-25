#include "impl/blas/cpu/lapack.h"
#include <stdexcept>
#include "lapack_wrap.hpp"
namespace lahva{

    namespace cpu
    {
        void SolveGenSysLinEquations(const char* T, const int n_, double* a, const int nrhs_, double* b)
        {
            LPCK_INT info = 0;
            LPCK_INT n = n_;
            LPCK_INT nrhs = nrhs_;
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

        void SolveGenSysLinEquations(const int n_, double* a, const int nrhs_, double* b, const char* T)
        {
            LPCK_INT info = 0;
            LPCK_INT n = n_;
            LPCK_INT nrhs = nrhs_;
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


        void SolveGenSysLinEquations(const char* T, const int n_, float* a, const int nrhs_, float* b)
        {
            LPCK_INT info = 0;
            LPCK_INT n = n_;
            LPCK_INT nrhs = nrhs_;
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

        void SolveGenSysLinEquations(const int n_, float* a, const int nrhs_, float* b, const char* T)
        {
            LPCK_INT info = 0;
            LPCK_INT n = n_;
            LPCK_INT nrhs = nrhs_;
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
