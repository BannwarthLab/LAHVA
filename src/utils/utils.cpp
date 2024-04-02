#include "const.h"
#include "utils.hpp"
#include <cstring>

namespace tcgmtensor
{
    BLAS_INT get_leading(BLAS_INT nrow, BLAS_INT ncol) {
    if (major == CblasColMajor) 
    {
        return std::max((BLAS_INT) 1, ncol);
    }
    else 
    {
        return std::max((BLAS_INT) 1, nrow);
    }    
    };

    CBLAS_TRANSPOSE get_trans(const char* T){
        if (std::strcmp(T,"T") or std::strcmp(T,"t"))
        {
            return CblasTrans;
        }
        else if (std::strcmp(T,"C") or std::strcmp(T,"c")) 
        {
            return CblasConjTrans;
        } 
        else 
        {
            return CblasNoTrans;
        }
    };

} // namespace tcgmtensor
