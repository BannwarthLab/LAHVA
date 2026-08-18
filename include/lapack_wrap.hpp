#ifndef LAHVA_LAPACK_WRAP_HPP
#define LAHVA_LAPACK_WRAP_HPP

#include <complex>
#ifdef _WIN32
    #define lapack_complex_double std::complex<double>
    #define lapack_complex_float std::complex<float>
    typedef unsigned int uint;
#endif
#if !defined(W_MKL) && !defined(_APPLE)
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
#ifdef _APPLE
#define ACCELERATE_NEW_LAPACK
typedef int64_t BLAS_INT;
#include <Accelerate/Accelerate.h>

typedef __LAPACK_int LPCK_INT;

typedef CBLAS_ORDER CBLAS_LAYOUT;
#endif
#ifdef W_MKL
#include <mkl.h>
typedef lapack_int LPCK_INT;
#endif

#endif // LAHVA_LAPACK_WRAP_HPP