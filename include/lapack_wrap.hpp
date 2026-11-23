/// @file lapack_wrap.hpp
/// @brief LAPACK library wrapper with platform-specific type definitions and includes.
///
/// This header provides a unified interface to different LAPACK implementations
/// (MKL, Accelerate, OpenBLAS, Netlib) by conditionally including the appropriate
/// headers and defining consistent integer type aliases (LPCK_INT) used throughout
/// the library for LAPACK function calls.

#pragma once

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
#include <Accelerate/Accelerate.h>

typedef CBLAS_ORDER CBLAS_LAYOUT;
#endif
#ifdef W_MKL
#include <mkl.h>
typedef lapack_int LPCK_INT;
#endif
