#include "const.h"
#include "runtime.hpp"
#include "linalg.hpp"

namespace tcgmtensor{

    namespace gpu
    {
        template<typename T>
        void SolveGenSysLinEquations(CudaRuntime& cudart, const char* Ta, Matrix<T>& a, Matrix<T>& b);
        template<typename T>
        void SolveGenSysLinEquations(CudaRuntime& cudart, Matrix<T>& a, Matrix<T>& b, const char* Ta = "N");
        
    } // namespace cpu
}