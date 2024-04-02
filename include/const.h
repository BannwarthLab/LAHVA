#ifndef TCGMBLAS_CONST_H
#define TCGMBLAS_CONST_H
#ifdef W_MKL
    #include <mkl.h>
    typedef MKL_INT BLAS_INT;
#else
    #include <cblas.h>
    typedef size_t BLAS_INT;
#endif

namespace tcgmtensor{
    static const CBLAS_LAYOUT major = CblasColMajor;
    static const CBLAS_UPLO tri = CblasLower;
}
#endif