#ifndef LAHVA_CONST_H
#define LAHVA_CONST_H
#include <stdint.h>
#ifdef W_MKL
#include <mkl.h>
typedef MKL_INT BLAS_INT;
typedef lapack_int LPCK_INT;
#elif _APPLE
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
typedef CBLAS_ORDER CBLAS_LAYOUT;
typedef int32_t BLAS_INT;
typedef int32_t LPCK_INT; 
#else
#include <cblas.h>
    #ifdef OPENBLAS_GENERIC
    typedef blasint BLAS_INT;
    #else
    typedef int32_t BLAS_INT;
    #endif
#ifdef NETLIB_HEADER
#include <netlib/lapack.h>
#else
#include <lapack.h>
#endif
    #ifdef OPENBLAS_GENERIC
    typedef blasint LPCK_INT;
    #else
    typedef int32_t LPCK_INT;
    #endif
#endif
#include <omp.h>
#include <climits>
#include <utility>
#include <cmath>
#ifdef _CUDA
#include "cublas_v2.h"
#endif

namespace lahva
{
    static const CBLAS_LAYOUT major = CblasColMajor;
    static const CBLAS_UPLO tri = CblasLower;
    static const int l_major = CblasColMajor;
    static char l_uplo = 'L';
    static char l_nondiag = 'N';
    static const char major_char = 'C';

    using Shape = std::pair<uint, uint>; 

    class BLASLeft
    {
    public:
        operator CBLAS_SIDE() { return CblasLeft; };
#ifdef _CUDA
        operator cublasSideMode_t() { return CUBLAS_SIDE_LEFT; }
#endif
    };

    class BLASRight
    {
    public:
        operator CBLAS_SIDE() { return CblasRight; };
#ifdef _CUDA
        operator cublasSideMode_t() { return CUBLAS_SIDE_RIGHT; }
#endif
    };

    class BLASUpper
    {
    public:
        operator CBLAS_UPLO() { return CblasUpper; };
#ifdef _CUDA
        operator cublasFillMode_t() { return CUBLAS_FILL_MODE_UPPER; }
#endif
    };

    class BLASLower
    {
    public:
        operator CBLAS_UPLO() { return CblasLower; };
#ifdef _CUDA
        operator cublasFillMode_t() { return CUBLAS_FILL_MODE_LOWER; }
#endif
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
