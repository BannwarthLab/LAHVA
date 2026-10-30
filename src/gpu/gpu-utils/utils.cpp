/// @file utils.cpp
/// @brief Implementation of GPU utility functions for BLAS operations.
///
/// Implements helper functions for managing GPU memory, BLAS operation parameters,
/// and tensor dimension validation.

#include <string>
#include "runtime.hpp"
#include "impl/gpu/utils.hpp"

namespace lahva {
    namespace gpu
    {
        /// @brief Calculates leading dimension for column-major matrix storage.
        ///
        /// Computes the leading dimension (row stride) for cuBLAS operations considering
        /// optional matrix transposition for proper column-major memory layout.
        ///
        /// @param nrow Number of rows.
        /// @param ncol Number of columns.
        /// @param transa Optional transpose operation (default: CUBLAS_OP_N for no transpose).
        /// @return Leading dimension value for column-major storage.
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
        
        /// @brief Converts character transposition flag to cuBLAS operation type.
        ///
        /// Translates string transposition indicators ('N', 'T', 'C') to corresponding
        /// cuBLAS operation types for use in BLAS function calls.
        ///
        /// @param T Transposition character: 'N' (no transpose), 'T' (transpose), 'C' (conjugate transpose).
        /// @return cuBLAS operation type (CUBLAS_OP_N, CUBLAS_OP_T, or CUBLAS_OP_C).
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

        /// @brief Converts character transposition flag to cuSPARSE operation type.
        ///
        /// Translates string transposition indicators ('N', 'T', 'C') to corresponding
        /// cuSPARSE operation types for use in sparse matrix operations. The sparse routines
        /// only support real types, for which 'C' is the same operation as 'T'.
        ///
        /// @param T Transposition character: 'N' (no transpose), 'T' (transpose), 'C' (conjugate transpose).
        /// @return cuSPARSE operation type (CUSPARSE_OPERATION_NON_TRANSPOSE or CUSPARSE_OPERATION_TRANSPOSE).
        cusparseOperation_t get_cusparse_trans(const char* T){
        if (std::strcmp(T,"T") == 0 or std::strcmp(T,"t") == 0 or std::strcmp(T,"C") == 0 or std::strcmp(T,"c") == 0)
        {
            return CUSPARSE_OPERATION_TRANSPOSE;
        }
        else
        {
            return CUSPARSE_OPERATION_NON_TRANSPOSE;
        }
    };

        /// @brief Flips the transposition flag for cuSPARSE operations.
        ///
        /// Converts between transpose and non-transpose operations for use in sparse matrix operations.
        ///
        /// @param op cuSPARSE operation type to flip.
        /// @return Flipped cuSPARSE operation type.
        cusparseOperation_t flip_cusparse_trans(cusparseOperation_t op){
        if (op == CUSPARSE_OPERATION_TRANSPOSE)
        {
            return CUSPARSE_OPERATION_NON_TRANSPOSE;
        }
        else
        {
            return CUSPARSE_OPERATION_TRANSPOSE;
        }
    };

        /// @brief Block shapes of C = op(A) * op(B) for block-diagonal matrices A and B.
        ///
        /// Use the result to construct an output matrix for the block-diagonal MatrixMatrixProduct.
        ///
        /// @tparam T Numeric element type (double or float).
        /// @param Ta Operation on A: "N" (no transpose), "T" (transpose) or "C" (conjugate transpose).
        /// @param Tb Operation on B: "N" (no transpose), "T" (transpose) or "C" (conjugate transpose).
        /// @param a Input block-diagonal matrix.
        /// @param b Input block-diagonal matrix.
        /// @return Shape of each block of C, in block order.
        /// @throws std::invalid_argument if the blocks of op(A) and op(B) cannot be multiplied.
        template <typename T>
        std::vector<Shape> product_block_shapes(const char *Ta, const char *Tb, const BlockDiagMatrix<T> &a, const BlockDiagMatrix<T> &b)
        {
            const bool trans_a = get_trans(Ta) != CUBLAS_OP_N;
            const bool trans_b = get_trans(Tb) != CUBLAS_OP_N;

            const size_t num_blocks = a.num_blocks();
            if (b.num_blocks() != num_blocks) {
                throw std::invalid_argument("product_block_shapes: block-diagonal matrices must have the same number of blocks");
            }

            // Looking up a shape by index walks the block map, so get them all at once
            const std::vector<size_t> a_rows = a.get_block_rows(), a_cols = a.get_block_cols();
            const std::vector<size_t> b_rows = b.get_block_rows(), b_cols = b.get_block_cols();
            const std::vector<size_t> &op_a_rows = trans_a ? a_cols : a_rows;
            const std::vector<size_t> &op_a_cols = trans_a ? a_rows : a_cols;
            const std::vector<size_t> &op_b_rows = trans_b ? b_cols : b_rows;
            const std::vector<size_t> &op_b_cols = trans_b ? b_rows : b_cols;

            if (op_a_cols != op_b_rows) {
                throw std::invalid_argument("product_block_shapes: blocks of op(A) and op(B) cannot be multiplied");
            }

            std::vector<Shape> shapes;
            shapes.reserve(num_blocks);
            for (size_t i = 0; i < num_blocks; ++i) {
                shapes.push_back(Shape{op_a_rows[i], op_b_cols[i]});
            }
            return shapes;
        }

        template std::vector<Shape> product_block_shapes(const char *Ta, const char *Tb, const BlockDiagMatrix<double> &a, const BlockDiagMatrix<double> &b);
        template std::vector<Shape> product_block_shapes(const char *Ta, const char *Tb, const BlockDiagMatrix<float> &a, const BlockDiagMatrix<float> &b);

    } // namespace gpu
} // namespace lahva
