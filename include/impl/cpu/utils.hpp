/// @file utils.hpp
/// @brief Public CPU utility functions for working with tensors.
///
/// Provides helper functions that are useful when calling the CPU BLAS routines. The internal
/// helpers of those routines, such as shape validation, are in src/cpu/utils/utils.hpp.

#pragma once
#include <vector>
#include "linalg.hpp"

namespace lahva
{
    namespace cpu
    {
        /// @brief Block shapes of C = op(A) * op(B) for block-diagonal matrices A and B.
        ///
        /// Use the result to construct an output matrix for the block-diagonal MatrixMatrixProduct.
        ///
        /// @tparam T Numeric element type (double or float).
        /// @param Ta Transpose for A: "N" (none), "T" (transpose), "C" (conjugate-transpose).
        /// @param Tb Transpose for B: "N" (none), "T" (transpose), "C" (conjugate-transpose).
        /// @param a  Block-diagonal input matrix.
        /// @param b  Block-diagonal input matrix.
        /// @return Shape of each block of C, in block order.
        /// @throws std::invalid_argument if the blocks of op(A) and op(B) cannot be multiplied.
        template <typename T>
        std::vector<Shape> product_block_shapes(const char *Ta, const char *Tb, const BlockDiagMatrix_<T> &a, const BlockDiagMatrix_<T> &b);

    } // namespace cpu
} // namespace lahva
