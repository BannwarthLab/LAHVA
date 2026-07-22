/// @file utils.cpp
/// @brief CPU utility functions for BLAS/LAPACK operations.
///
/// Provides helper functions for calculating leading dimensions in column-major storage,
/// handling transpose flags, shape validation, and other utilities supporting CPU BLAS/LAPACK
/// functions throughout the library.

#include <cstring>
#include <iostream>
#include "const.h"
#include "utils.hpp"

namespace lahva
{
    namespace cpu
    {

        BLAS_INT get_leading(BLAS_INT nrow, BLAS_INT ncol, CBLAS_TRANSPOSE trans)
        {
            if (major == CblasColMajor)
            {
                if (trans == CblasNoTrans)
                {
                    return std::max((BLAS_INT)1, nrow);
                }
                else
                {
                    return std::max((BLAS_INT)1, ncol);
                }
            }
            else
            {
                if (trans == CblasNoTrans)
                {
                    return std::max((BLAS_INT)1, ncol);
                }
                else
                {
                    return std::max((BLAS_INT)1, nrow);
                }
            }
        };

        CBLAS_TRANSPOSE get_trans(const char *T)
        {
            if ((std::strcmp(T, "T") == 0) || (std::strcmp(T, "t") == 0))
            {
                return CblasTrans;
            }
            else if ((std::strcmp(T, "C") == 0) || (std::strcmp(T, "c") == 0))
            {
                return CblasConjTrans;
            }
            else
            {
                return CblasNoTrans;
            }
        };
    }
} // namespace lahva
