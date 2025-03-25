#include "const.h"
namespace lahva{

    namespace cpu
    {
        void SolveGenSysLinEquations(const char* T, const int n, double* a, const int nrhs, double* b);
        void SolveGenSysLinEquations(const int n, double* a, const int nrhs, double* b, const char* T = "N");
        void SolveGenSysLinEquations(const char* T, const int n, float* a, const int nrhs, float* b);
        void SolveGenSysLinEquations(const int n, float* a, const int nrhs, float* b, const char* T = "N");
    } // namespace cpu
    
}