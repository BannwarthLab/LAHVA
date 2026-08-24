/// @file utils.cpp
/// @brief Implementation of GPU utility functions for BLAS operations.
///
/// Implements helper functions for managing GPU memory, BLAS operation parameters,
/// and tensor dimension validation.

#include <string>
#include "runtime.hpp"
#include "impl/gpu/utils.hpp"

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

        /// @brief Converts character transposition flag to cuSPARSE operation type.
        ///
        /// Translates string transposition indicators ('N', 'T') to corresponding
        /// cuSPARSE operation types for use in sparse matrix operations.
        ///
        /// @param T Transposition character: 'N' (no transpose), 'T' (transpose).
        /// @return cuSPARSE operation type (CUSPARSE_OPERATION_NON_TRANSPOSE or CUSPARSE_OPERATION_TRANSPOSE).
        cusparseOperation_t get_cusparse_trans(const char* T){
        if (std::strcmp(T,"T") == 0 or std::strcmp(T,"t") == 0)
        {
            return CUSPARSE_OPERATION_TRANSPOSE;
        }
        else
        {
            return CUSPARSE_OPERATION_NON_TRANSPOSE;
        }
    };

        /// @brief Flips the transposition flag for cuSPARSE operations.
        ///
        /// Converts between transpose and non-transpose operations for use in sparse matrix operations.
        ///
        /// @param op cuSPARSE operation type to flip.
        /// @return Flipped cuSPARSE operation type.
        cusparseOperation_t flip_cusparse_trans(cusparseOperation_t op){
        if (op == CUSPARSE_OPERATION_TRANSPOSE)
        {
            return CUSPARSE_OPERATION_NON_TRANSPOSE;
        }
        else
        {
            return CUSPARSE_OPERATION_TRANSPOSE;
        }
    };

    } // namespace gpu
} // namespace lahva
