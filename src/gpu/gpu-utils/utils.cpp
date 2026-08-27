#include "utils.hpp"
#include "runtime.hpp"
#include <string>

namespace lahva {
    namespace gpu
    {
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
    
}
