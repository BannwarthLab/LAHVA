#ifndef TCGMBLAS_CONST_H
#define TCGMBLAS_CONST_H
#ifdef W_MKL
#include <mkl.h>
typedef MKL_INT BLAS_INT;
typedef lapack_int LPCK_INT;
#else
#include <cblas.h>
#include <lapacke.h>
typedef size_t BLAS_INT;
typedef lapck_int LCPK_INT;
#endif
#include <omp.h>
#ifdef _CUDA
#pragma warning(disable : 2282)
#include "cublas_v2.h"
#endif

namespace tcgmtensor
{
    static const CBLAS_LAYOUT major = CblasColMajor;
    static const CBLAS_UPLO tri = CblasLower;
    static const int l_major = LAPACK_COL_MAJOR;
    static char l_uplo = 'L'; 

    class BLASLeft
    {
    public:
        operator CBLAS_SIDE() { return CblasLeft; };
        operator cublasSideMode_t() { return CUBLAS_SIDE_LEFT; }
    };

    class BLASRight
    {
    public:
        operator CBLAS_SIDE() { return CblasRight; };
        operator cublasSideMode_t() { return CUBLAS_SIDE_RIGHT; }
    };

    class BLASUpper
    {
    public:
        operator CBLAS_UPLO() { return CblasUpper; };
        operator cublasFillMode_t() { return CUBLAS_FILL_MODE_UPPER; }
    };

    class BLASLower
    {
    public:
        operator CBLAS_UPLO() { return CblasLower; };
        operator cublasFillMode_t() { return CUBLAS_FILL_MODE_LOWER; }
    };

    class BLASRuntime
    {
    public:
        BLASLeft left;
        BLASRight right;
        BLASUpper upper;
        BLASLower lower;
        BLASRuntime(){};
        virtual ~BLASRuntime(){};
    };

    class CPURuntime : public BLASRuntime
    {
    };

#ifdef _CUDA
    static const cublasFillMode_t tri_gpu = CUBLAS_FILL_MODE_LOWER;
#endif
}
#endif