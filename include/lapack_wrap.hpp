#ifndef LAHVA_LAPACK_WRAP_HPP
#define LAHVA_LAPACK_WRAP_HPP
#if not defined(ACCELERATE_NEW_LAPACK) and not defined(W_MKL)
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
#endif // LAHVA_LAPACK_WRAP_HPP