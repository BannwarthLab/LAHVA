/// @file additional-level1.cuh
/// @brief GPU kernel for applying unary operations to tensors.
///
/// Provides a generic GPU kernel template for applying custom unary operations
/// element-wise to tensors.

#pragma once
#include "runtime.hpp"
namespace lahva
{
    namespace gpu
    {
        /// @brief GPU kernel to apply unary operation element-wise to tensor.
        ///
        /// Applies a unary operation func to each element in-place: a[i] = func(a[i]).
        /// Uses grid-stride loop for flexible grid size.
        ///
        /// @tparam T Element type.
        /// @tparam op Unary operation functor type.
        /// @param ndim Total number of elements.
        /// @param a Input/output tensor (modified in-place).
        /// @param func Unary operation functor to apply to each element.
        template<typename T, class op>
        __global__ void ApplyKernel_(unsigned long long ndim, T *a, op func)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < ndim)
            {
                a[index] = func(a[index]);
            }
        };

    } // namespace gpu
    
} // namespace lahva