/// @file utils.cpp
/// @brief CPU utility functions for BLAS/LAPACK operations.
///
/// Provides helper functions for calculating leading dimensions in column-major storage,
/// handling transpose flags, shape validation, and other utilities supporting CPU BLAS/LAPACK
/// functions throughout the library.

#include <cstring>
#include <iostream>
#include <stdexcept>
#include "const.h"
#include "impl/cpu/utils.hpp"
#include "utils.hpp"

namespace lahva
{
    namespace cpu
    {

        /// @brief Compute the leading dimension for BLAS column-major storage.
        ///
        /// Determines the leading dimension (memory stride between columns) for a matrix
        /// stored in column-major order. For non-transposed matrices, this is the number
        /// of rows. For transposed matrices, this is the number of columns.
        ///
        /// @param nrow Number of rows in the original matrix.
        /// @param ncol Number of columns in the original matrix.
        /// @param trans Transpose flag indicating the actual storage layout (default: CblasNoTrans).
        /// @return Leading dimension value for use with CBLAS routines.
        BLAS_INT get_leading(BLAS_INT nrow, BLAS_INT ncol, CBLAS_TRANSPOSE trans)
        {
            if (major == CblasColMajor)
            {
                if (trans == CblasNoTrans)
                {
                    return std::max((BLAS_INT)1, nrow);
                }
                else
                {
                    return std::max((BLAS_INT)1, ncol);
                }
            }
            else
            {
                if (trans == CblasNoTrans)
                {
                    return std::max((BLAS_INT)1, ncol);
                }
                else
                {
                    return std::max((BLAS_INT)1, nrow);
                }
            }
        };

        /// @brief Convert character transpose flag to CBLAS_TRANSPOSE enumeration.
        ///
        /// Translates a character representation of transpose operation to the corresponding
        /// CBLAS_TRANSPOSE enum value: "N" or "n" -> CblasNoTrans, "T" or "t" -> CblasTrans, 
        /// "C" or "c" -> CblasConjTrans. Case-insensitive.
        ///
        /// @param T Character string "N" (no transpose), "T" (transpose), or "C" (conjugate transpose).
        /// @return Corresponding CBLAS_TRANSPOSE enumeration value.
        CBLAS_TRANSPOSE get_trans(const char *T)
        {
            if ((std::strcmp(T, "T") == 0) || (std::strcmp(T, "t") == 0))
            {
                return CblasTrans;
            }
            else if ((std::strcmp(T, "C") == 0) || (std::strcmp(T, "c") == 0))
            {
                return CblasConjTrans;
            }
            else
            {
                return CblasNoTrans;
            }
        };

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
        std::vector<Shape> product_block_shapes(const char *Ta, const char *Tb, const BlockDiagMatrix_<T> &a, const BlockDiagMatrix_<T> &b)
        {
            const bool trans_a = get_trans(Ta) != CblasNoTrans;
            const bool trans_b = get_trans(Tb) != CblasNoTrans;

            const size_t num_blocks = a.num_blocks();
            if (b.num_blocks() != num_blocks) {
                throw std::invalid_argument("product_block_shapes: block-diagonal matrices must have the same number of blocks");
            }

            std::vector<Shape> shapes;
            shapes.reserve(num_blocks);
            for (size_t i = 0; i < num_blocks; ++i) {
                const Shape sa = a.get_block_shape(i);
                const Shape sb = b.get_block_shape(i);
                const size_t op_a_rows = trans_a ? sa.second : sa.first;
                const size_t op_a_cols = trans_a ? sa.first : sa.second;
                const size_t op_b_rows = trans_b ? sb.second : sb.first;
                const size_t op_b_cols = trans_b ? sb.first : sb.second;
                if (op_a_cols != op_b_rows) {
                    throw std::invalid_argument("product_block_shapes: blocks of op(A) and op(B) cannot be multiplied");
                }
                shapes.push_back(Shape{op_a_rows, op_b_cols});
            }
            return shapes;
        }

        template std::vector<Shape> product_block_shapes(const char *Ta, const char *Tb, const BlockDiagMatrix_<double> &a, const BlockDiagMatrix_<double> &b);
        template std::vector<Shape> product_block_shapes(const char *Ta, const char *Tb, const BlockDiagMatrix_<float> &a, const BlockDiagMatrix_<float> &b);
    }
} // namespace lahva
