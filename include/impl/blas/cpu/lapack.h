#include "const.h"
#include "lapack_wrap.hpp"
namespace lahva{

    namespace cpu
    {
        void SolveGenSysLinEquations(const char* T, const LPCK_INT n, double* a, const LPCK_INT nrhs, double* b);
        void SolveGenSysLinEquations(const LPCK_INT n, double* a, const LPCK_INT nrhs, double* b, const char* T = "N");
        void SolveGenSysLinEquations(const char* T, const LPCK_INT n, float* a, const LPCK_INT nrhs, float* b);
        void SolveGenSysLinEquations(const LPCK_INT n, float* a, const LPCK_INT nrhs, float* b, const char* T = "N");
    } // namespace cpu
    
}