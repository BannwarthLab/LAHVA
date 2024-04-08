#include "utils.hpp"
#include "runtime.hpp"
#include <string>

namespace tcgmtensor {
    namespace gpu
    {
        size_t get_leading(size_t nrow, size_t ncol) {
            return std::max((size_t) 1, nrow);
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
    } // namespace gpu
    
}
