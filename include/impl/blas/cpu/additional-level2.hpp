/// @file additional-level2.hpp
/// @brief C++-style additional Level-2 operations beyond standard BLAS.
///
// Additional Level-2 BLAS-like operations (matrix * matrix) - C++ template declarations
// These functions are C++-style wrappers providing additional matrix operations
// beyond standard BLAS Level-2 operations. These wrappers operate on Matrix_ objects and
// dispatch to optimized CPU implementations. Each overload is provided for double and
// float precision; both precisions are documented explicitly.

#pragma once
#include "linalg.hpp"
#include "const.h"

namespace lahva
{
    namespace cpu
    {

        
        template <typename T>
        void AddMatrices(const char *Ta, const char *Tb, const T alpha, const Matrix_<T> &a, const T beta,
                         const Matrix_<T> &b, Matrix_<T> &c);
        template <typename T>
        void AddMatrices(const Matrix_<T> &a, const Matrix_<T> &b, Matrix_<T> &c,
                         const T alpha = 1.0, const T beta = 1.0, const char *Ta = "N", const char *Tb = "N");

        template <typename... Args>
        void AddMatrices(const CPURuntime &rt_, Args &&...args)
        {
            (AddMatrices(args...));
        }

    } // namespace gpu

}