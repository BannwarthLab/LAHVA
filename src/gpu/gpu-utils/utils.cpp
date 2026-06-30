/// @file utils.cpp
/// @brief Implementation of GPU utility functions for BLAS operations.
///
/// Implements helper functions for managing GPU memory, BLAS operation parameters,
/// and tensor dimension validation.

#include "utils.hpp"
#include "runtime.hpp"
#include <string>

namespace lahva {
    namespace gpu
    {
        /// Implementation of get_leading - see utils.hpp for documentation.
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

        /// Implementation of get_trans - see utils.hpp for documentation.
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
    
}
