#include "utils.hpp"
#include "runtime.hpp"
/// @file utils.cpp
/// @brief Implementation of GPU utility functions for BLAS operations.
///
/// Implements helper functions for managing GPU memory, BLAS operation parameters,
/// and tensor dimension validation.

#include <string>

namespace lahva {
    namespace gpu
    {
        /// @brief Calculates leading dimension for column-major matrix storage.
        ///
        /// Computes the leading dimension (row stride) for cuBLAS operations considering
        /// optional matrix transposition for proper column-major memory layout.
        ///
        /// @param nrow Number of rows.
        /// @param ncol Number of columns.
        /// @param transa Optional transpose operation (default: CUBLAS_OP_N for no transpose).
        /// @return Leading dimension value for column-major storage.
        size_t get_leading(size_t nrow, size_t ncol, cublasOperation_t transa){ 

            if (transa == CUBLAS_OP_N)
            {
                return std::max((size_t) 1, nrow);
            }
            else
            {
                return std::max((size_t) 1, ncol);
            }

        };
        
        /// @brief Converts character transposition flag to cuBLAS operation type.
        ///
        /// Translates string transposition indicators ('N', 'T', 'C') to corresponding
        /// cuBLAS operation types for use in BLAS function calls.
        ///
        /// @param T Transposition character: 'N' (no transpose), 'T' (transpose), 'C' (conjugate transpose).
        /// @return cuBLAS operation type (CUBLAS_OP_N, CUBLAS_OP_T, or CUBLAS_OP_C).
        cublasOperation_t get_trans(const char* T){
        if (std::strcmp(T,"T") == 0 or std::strcmp(T,"t") == 0)
        {
            return CUBLAS_OP_T;
        }
        else if (std::strcmp(T,"C") == 0 or std::strcmp(T,"c") == 0) 
        {
            return CUBLAS_OP_C;
        } 
        else 
        {
            return CUBLAS_OP_N;
        }
    };
    } // namespace gpu

} // namespace lahva
