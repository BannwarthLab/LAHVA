#include "lapack_wrap.hpp"
#include "impl/blas/cpu/lapack.h"
#include <stdexcept>
namespace lahva
{

    namespace cpu
    {
        void SolveGenSysLinEquations(const char *T, const LPCK_INT n, double *a, const LPCK_INT nrhs, double *b)
        {
            LPCK_INT info = 0;
            LPCK_INT *ipiv = new LPCK_INT[n];
#ifdef _APPLE
            dgetrf_(&n, &n, a, &n, ipiv, &info);
#elif defined(W_MKL)
            info = LAPACKE_dgetrf(l_major, n, n, a, n, ipiv);
#else
            LAPACK_dgetrf(&n, &n, a, &n, ipiv, &info);
#endif
            if (info != 0)
            {
                throw std::runtime_error("Failure in DGETRF");
            }
#ifdef _APPLE
            dgetrs_(T, &n, &nrhs, a, &n, ipiv, b, &n, &info);
#elif  defined(W_MKL)
            info = LAPACKE_dgetrs(l_major, *T, n, nrhs, a, n, ipiv, b, n);
#else
            LAPACK_dgetrs(T, &n, &nrhs, a, &n, ipiv, b, &n, &info);
#endif
            if (info != 0)
            {
                throw std::runtime_error("Failure in DGETRS");
            }
            delete[] ipiv;
        };

        void SolveGenSysLinEquations(const LPCK_INT n, double *a, const LPCK_INT nrhs, double *b, const char *T="N")
        {
            LPCK_INT info = 0;
            LPCK_INT *ipiv = new LPCK_INT[n];

#ifdef _APPLE
            dgetrf_(&n, &n, a, &n, ipiv, &info);
#elif defined(W_MKL)
            info = LAPACKE_dgetrf(l_major, n, n, a, n, ipiv);
#else
            LAPACK_dgetrf(&n, &n, a, &n, ipiv, &info);
#endif
            if (info != 0)
            {
                throw std::runtime_error("Failure in DGETRF");
            }
#ifdef _APPLE
            dgetrs_(T, &n, &nrhs, a, &n, ipiv, b, &n, &info);
#elif  defined(W_MKL)
            info = LAPACKE_dgetrs(l_major, *T, n, nrhs, a, n, ipiv, b, n);
#else
            LAPACK_dgetrs(T, &n, &nrhs, a, &n, ipiv, b, &n, &info);
#endif
            if (info != 0)
            {
                throw std::runtime_error("Failure in DGETRS");
            }
            delete[] ipiv;
        };

        void SolveGenSysLinEquations(const char *T, const LPCK_INT n, float *a, const LPCK_INT nrhs, float *b)
        {
            LPCK_INT info = 0;
            LPCK_INT *ipiv = new LPCK_INT[n];
#ifdef _APPLE
            sgetrf_(&n, &n, a, &n, ipiv, &info);
#elif defined(W_MKL)
            info = LAPACKE_sgetrf(l_major, n, n, a, n, ipiv);
#else
            LAPACK_sgetrf(&n, &n, a, &n, ipiv, &info);
#endif
            if (info != 0)
            {
                throw std::runtime_error("Failure in SGETRF");
            }
#ifdef _APPLE
            sgetrs_(T, &n, &nrhs, a, &n, ipiv, b, &n, &info);
#elif defined(W_MKL)
            info = LAPACKE_sgetrs(l_major, *T, n, nrhs, a, n, ipiv, b, n);
#else
            LAPACK_sgetrs(T, &n, &nrhs, a, &n, ipiv, b, &n, &info);
#endif

            if (info != 0)
            {
                throw std::runtime_error("Failure in SGETRS");
            }
            delete[] ipiv;
        };

        void SolveGenSysLinEquations(const LPCK_INT n, float *a, const LPCK_INT nrhs, float *b, const char *T="N")
        {
            LPCK_INT info = 0;
            LPCK_INT *ipiv = new LPCK_INT[n];
#ifdef _APPLE
            sgetrf_(&n, &n, a, &n, ipiv, &info);
#elif defined(W_MKL)
            info = LAPACKE_sgetrf(l_major, n, n, a, n, ipiv);
#else
            LAPACK_sgetrf(&n, &n, a, &n, ipiv, &info);
#endif
            if (info != 0)
            {
                throw std::runtime_error("Failure in SGETRF");
            }
#ifdef _APPLE
            sgetrs_(T, &n, &nrhs, a, &n, ipiv, b, &n, &info);
#elif defined(W_MKL)
            info = LAPACKE_sgetrs(l_major, *T, n, nrhs, a, n, ipiv, b, n);
#else
            LAPACK_sgetrs(T, &n, &nrhs, a, &n, ipiv, b, &n, &info);
#endif
            if (info != 0)
            {
                throw std::runtime_error("Failure in SGETRS");
            }
            delete[] ipiv;
        };
    } // namespace cpu

}
