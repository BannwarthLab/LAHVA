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

        /// @brief Compute the leading dimension for BLAS column-major storage.
        ///
        /// Determines the leading dimension (memory stride between columns) for a matrix
        /// stored in column-major order. For non-transposed matrices, this is the number
        /// of rows. For transposed matrices, this is the number of columns.
        ///
        /// @param nrow Number of rows in the original matrix.
        /// @param ncol Number of columns in the original matrix.
        /// @param trans Transpose flag indicating the actual storage layout (default: CblasNoTrans).
        /// @return Leading dimension value for use with CBLAS routines.
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

        /// @brief Convert character transpose flag to CBLAS_TRANSPOSE enumeration.
        ///
        /// Translates a character representation of transpose operation to the corresponding
        /// CBLAS_TRANSPOSE enum value: "N" or "n" -> CblasNoTrans, "T" or "t" -> CblasTrans, 
        /// "C" or "c" -> CblasConjTrans. Case-insensitive.
        ///
        /// @param T Character string "N" (no transpose), "T" (transpose), or "C" (conjugate transpose).
        /// @return Corresponding CBLAS_TRANSPOSE enumeration value.
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
