#ifndef LAHVA_GPU_UTILS_HPP
#define LAHVA_GPU_UTILS_HPP
#include "linalg.hpp"
#include "runtime.hpp"

namespace lahva
{
    namespace gpu
    {

        

        template <typename T>
        void check_device_alloc(const CudaRuntime &cudart, const GPUTensor_<T> &gpu_vec)
        {
            
            gpu_vec.copy2device(cudart);

        };

   

        template <typename T>
        std::tuple<size_t, size_t> check_size_mv(const Matrix_<T> &m, const GPUTensor_<T> &vmult, const GPUTensor_<T> &vres, cublasOperation_t trans = CUBLAS_OP_N)
        {
            Shape s = m.shape();
            size_t nrow = s.first;
            size_t ncol = s.second;
            if (trans == CUBLAS_OP_N)
            {
                assert(nrow == vres.size());
                assert(ncol == vmult.size());
            }
            else
            {
                assert(nrow == vmult.size());
                assert(ncol == vres.size());
            }

            return std::make_tuple(nrow, ncol);
        };

        template <typename T>
        std::tuple<size_t, size_t> check_size_mv(const LowTriMatrix_<T> &m, const Vector_<T> &vmult, const Vector_<T> &vres, cublasOperation_t trans = CUBLAS_OP_N)
        {
            Shape s = m.shape();
            size_t nrow = s.first;
            size_t ncol = s.second;
            if (trans == CUBLAS_OP_N)
            {
                assert(nrow == vres.size());
                assert(ncol == vmult.size());
            }
            else
            {
                assert(nrow == vmult.size());
                assert(ncol == vres.size());
            }

            return std::make_tuple(nrow, ncol);
        };

        size_t get_leading(size_t nrow, size_t ncol, cublasOperation_t transa = CUBLAS_OP_N);

        cublasOperation_t get_trans(const char *T);

        /// @brief Converts character transposition flag to cuSPARSE operation type.
        ///
        /// Translates string transposition indicators ('N', 'T') to corresponding
        /// cuSPARSE operation types for use in sparse matrix operations.
        ///
        /// @param T Transposition character: 'N' (no transpose), 'T' (transpose).
        /// @return cuSPARSE operation type (CUSPARSE_OPERATION_NON_TRANSPOSE or CUSPARSE_OPERATION_TRANSPOSE).
        cusparseOperation_t get_cusparse_trans(const char *T);

        template <typename T, typename U, typename V>
        std::tuple<size_t, size_t, size_t> check_size_mm(const Matrix_<T> &a, const Matrix_<U> &b,
                                                         const Matrix_<V> &c, cublasOperation_t transa = CUBLAS_OP_N, cublasOperation_t transb = CUBLAS_OP_N)
        {

            Shape sa = a.shape();
            size_t nrowa = sa.first;
            size_t ncola = sa.second;
#ifndef NDEBUG
            Shape sb = b.shape();
            size_t nrowb = sb.first;
            size_t ncolb = sb.second;
#endif
            Shape sc = c.shape();
            size_t nrowc = sc.first;
            size_t ncolc = sc.second;

            size_t k;

            if (transa == CUBLAS_OP_N)
            {
                if (transb == CUBLAS_OP_N)
                {
                    assert(ncola == nrowb);
                    assert(nrowa == nrowc);
                    assert(ncolb == ncolc);
                    k = ncola;
                }
                else // B is transposed
                {
                    assert(ncola == ncolb);
                    assert(nrowa == nrowc);
                    assert(nrowb == ncolc);
                    k = ncola;
                }
            }
            else // A is transposed
            {
                if (transb == CUBLAS_OP_N)
                {
                    assert(nrowa == nrowb);
                    assert(ncola == nrowc);
                    assert(ncolb == ncolc);
                    k = nrowa;
                }
                else // A and B are transposed
                {
                    assert(nrowa == ncolb);
                    assert(ncola == nrowc);
                    assert(nrowb == ncolc);
                    k = ncola;
                }
            }

            return std::make_tuple(nrowc, ncolc, k);
        };

        template<typename T>
    std::tuple<size_t, size_t> check_same_shape_mm(const Matrix_<T>& a, const Matrix_<T>& b, 
        const Matrix_<T>& c, cublasOperation_t transa = CUBLAS_OP_N, cublasOperation_t transb = CUBLAS_OP_N){
#ifndef NDEBUG
        Shape sa = a.shape();
        size_t nrowa = sa.first;
        size_t ncola = sa.second;

        Shape sb = b.shape();
        size_t nrowb = sb.first;
        size_t ncolb = sb.second;
#endif
        Shape sc = c.shape();
        size_t nrowc = sc.first;
        size_t ncolc = sc.second;

        if (transa == CUBLAS_OP_N) {
            if (transb == CUBLAS_OP_N)
            {
                assert(ncola == ncolb);
                assert(nrowa == nrowb);
                assert(ncolb == ncolc);
                assert(nrowb == nrowc);
            }
            else // B is transposed
            {
                assert(ncola == nrowb);
                assert(nrowa == ncolb);
                assert(ncola == ncolc);
                assert(nrowa == nrowc);
            }   
        }
        else // A is transposed 
        {
           if (transb == CUBLAS_OP_N)
            {
                assert(nrowa == ncolb);
                assert(ncola == nrowb);
                assert(ncolb == ncolc);
                assert(nrowb == nrowc);
            }
            else // A and B are transposed
            {
                assert(nrowa == nrowb);
                assert(ncola == ncolb);
                assert(ncolb == nrowc);
                assert(nrowa == ncolc);
            }    
        }
        

        return std::make_tuple(nrowc, ncolc);
    };

 /// @brief Validates tensor shapes for block matrix-vector multiplication with cuSPARSE operation type.
        ///
        /// Checks that the block matrix and vector dimensions are compatible for sparse matrix-vector product.
        /// Assertions verify that dimensions match considering optional transposition of the matrix.
        ///
        /// @tparam T Tensor element type.
        /// @param m Input block matrix.
        /// @param vmult Vector to multiply (must match matrix column count or row count if transposed).
        /// @param vres Result vector (must match matrix row count or column count if transposed).
        /// @param trans Optional transpose operation on matrix (default: CUSPARSE_OPERATION_NON_TRANSPOSE for no transpose).
        /// @return Tuple of (nrow, ncol) dimensions of the matrix.
        template <typename T>
        std::tuple<size_t, size_t> check_size_mv(const BlockMatrix_<T> &m, const GPUTensor_<T> &vmult, const GPUTensor_<T> &vres, cusparseOperation_t trans)
        {
            Shape s = m.shape();
            size_t nrow = s.first;
            size_t ncol = s.second;
            if (trans == CUSPARSE_OPERATION_NON_TRANSPOSE)
            {
                assert(nrow == vres.size());
                assert(ncol == vmult.size());
            }
            else
            {
                assert(nrow == vmult.size());
                assert(ncol == vres.size());
            }

            return std::make_tuple(nrow, ncol);
        };

        /// @brief Converts C++ type to corresponding CUDA data type.
        ///
        /// Maps template type T to the appropriate cudaDataType_t for use in cuBLAS/cuSPARSE operations.
        ///
        /// @tparam T Numeric element type (float, double, or complex types).
        /// @return CUDA data type constant (CUDA_R_32F, CUDA_R_64F, CUDA_C_32F, or CUDA_C_64F).
        /// @throws std::invalid_argument if type T is not supported.
        template <typename T>
        cudaDataType_t get_cuda_datatype()
        {
            if constexpr (std::is_same_v<T, float>)
            {
                return CUDA_R_32F;
            }
            else if constexpr (std::is_same_v<T, double>)
            {
                return CUDA_R_64F;
            }
            else if constexpr (std::is_same_v<T, complex_float>)
            {
                return CUDA_C_32F;
            }
            else if constexpr (std::is_same_v<T, complex_double>)
            {
                return CUDA_C_64F;
            }
            else
            {
                throw std::invalid_argument("get_cuda_datatype: unsupported type");
            }
        };

        /// @brief Flips the transposition flag for cuSPARSE operations.
        ///
        /// Converts between transpose and non-transpose operations for use in sparse matrix operations.
        ///
        /// @param op cuSPARSE operation type to flip.
        /// @return Flipped cuSPARSE operation type.
        cusparseOperation_t flip_cusparse_trans(cusparseOperation_t op);

        /// @brief Validates dimensions for sparse-dense matrix multiplication (sparse A * dense B).
    ///
    /// Checks that sparse matrix A and dense matrix B have compatible dimensions for
    /// C = alpha*op(A)*op(B) + beta*C operation, where A is a BlockMatrix.
    /// Assertions verify dimension compatibility considering optional transposition.
    ///
    /// @tparam T Element type for matrices.
    /// @param sparse_a Sparse block matrix (first operand).
    /// @param dense_b Dense matrix (second operand).
    /// @param OpA Sparse operation on A (CUSPARSE_OPERATION_NON_TRANSPOSE or CUSPARSE_OPERATION_TRANSPOSE).
    /// @param OpB Sparse operation on B (CUSPARSE_OPERATION_NON_TRANSPOSE or CUSPARSE_OPERATION_TRANSPOSE).
    /// @return Tuple of (M, N, m, k) where:
    ///         - M, N are result matrix dimensions
    ///         - m, k are original sparse matrix dimensions
    template <typename T>
    std::tuple<int64_t, int64_t, int64_t, int64_t> check_size_sparse_dense(
        const BlockMatrix_<T> &sparse_a,
        const Matrix_<T> &dense_b,
        cusparseOperation_t OpA,
        cusparseOperation_t OpB)
    {
        int64_t rows_A = static_cast<int64_t>(sparse_a.shape().first);
        int64_t cols_A = static_cast<int64_t>(sparse_a.shape().second);
        int64_t rows_B = static_cast<int64_t>(dense_b.shape().first);
        int64_t cols_B = static_cast<int64_t>(dense_b.shape().second);

        bool transA = (OpA == CUSPARSE_OPERATION_TRANSPOSE);
        bool transB = (OpB == CUSPARSE_OPERATION_TRANSPOSE);

        int64_t M = transA ? cols_A : rows_A;
        int64_t N = transB ? rows_B : cols_B;
        int64_t m = rows_A;
        int64_t k = cols_A;

        return std::make_tuple(M, N, m, k);
    };

    /// @brief Validates dimensions for dense-sparse matrix multiplication (dense A * sparse B).
    ///
    /// Checks that dense matrix A and sparse matrix B have compatible dimensions for
    /// C = alpha*op(A)*op(B) + beta*C operation, where B is a BlockMatrix.
    /// Assertions verify dimension compatibility considering optional transposition.
    ///
    /// @tparam T Element type for matrices.
    /// @param dense_a Dense matrix (first operand).
    /// @param sparse_b Sparse block matrix (second operand).
    /// @param OpA Operation on A (CUSPARSE_OPERATION_NON_TRANSPOSE or CUSPARSE_OPERATION_TRANSPOSE).
    /// @param OpB Operation on B (CUSPARSE_OPERATION_NON_TRANSPOSE or CUSPARSE_OPERATION_TRANSPOSE).
    /// @return Tuple of (M, N, m, k) where:
    ///         - M, N are result matrix dimensions
    ///         - m, k are original sparse matrix dimensions
    template <typename T>
    std::tuple<int64_t, int64_t, int64_t, int64_t> check_size_dense_sparse(
        const Matrix_<T> &dense_a,
        const BlockMatrix_<T> &sparse_b,
        cusparseOperation_t OpA,
        cusparseOperation_t OpB)
    {
        int64_t rows_A = static_cast<int64_t>(dense_a.shape().first);
        int64_t cols_A = static_cast<int64_t>(dense_a.shape().second);
        int64_t rows_B = static_cast<int64_t>(sparse_b.shape().first);
        int64_t cols_B = static_cast<int64_t>(sparse_b.shape().second);

        bool transA = (OpA == CUSPARSE_OPERATION_TRANSPOSE);
        bool transB = (OpB == CUSPARSE_OPERATION_TRANSPOSE);

        int64_t M = transA ? cols_A : rows_A;
        int64_t N = transB ? rows_B : cols_B;
        int64_t m = rows_B;
        int64_t k = cols_B;

        return std::make_tuple(M, N, m, k);
    };

        /// @brief Unpacks block vector from padded to scattered layout on device.
        ///
        /// Launches GPU kernel to redistribute padded block vector back to original layout.
        /// Used after batched block operations to restore vector layout for further computation.
        /// Each block is stored contiguously in padded format with fixed size `max_size` on the GPU,
        /// and is scattered back to its original position based on offset and block size information.
        ///
        /// @tparam T Floating point type (float or double).
        /// @param[in] cudart CUDA runtime context for stream and block size configuration.
        /// @param[in] src Device pointer to padded block vector [num_blocks * max_size].
        /// @param[out] dst Device pointer to output vector in scattered layout [total_size].
        /// @param[in] d_offsets Device array of starting positions for each block [num_blocks].
        /// @param[in] d_block_sizes Device array of actual block sizes [num_blocks].
        /// @param[in] max_size Maximum block size (padding size for uniform blocks).
        /// @param[in] num_blocks Number of blocks to unpack.
        template <typename T>
        void unpack_vector(
            const CudaRuntime &cudart,
            const T *src,
            T *dst,
            const int *d_offsets,
            const int *d_block_sizes,
            size_t max_size,
            int num_blocks);




    } // namespace gpu

}
#endif