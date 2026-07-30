/// @file utils.hpp
/// @brief GPU utility functions for device memory management and BLAS operation validation.
///
/// Provides helper functions for managing GPU tensor allocation, validating tensor shapes
/// for BLAS operations, and managing CUDA operations like stream handling.

#pragma once
#include "linalg.hpp"
#include "runtime.hpp"

namespace lahva
{
    namespace gpu
    {
        /// @brief Ensures GPU tensor data is allocated and copied to device if needed.
        ///
        /// Calls the tensor's copy2device method through the CUDA runtime to ensure
        /// all data is properly transferred to GPU device memory.
        ///
        /// @tparam T Tensor element type.
        /// @param cudart CUDA runtime instance.
        /// @param gpu_vec GPU tensor to check and allocate if needed.
        template <typename T>
        void check_device_alloc(const CudaRuntime &cudart, const Tensor_<T> &gpu_vec)
        {
            
            gpu_vec.copy2device(cudart);

        };

   

        /// @brief Validates tensor shapes for matrix-vector multiplication operation.
        ///
        /// Checks that the matrix and vector dimensions are compatible for matrix-vector product.
        /// Assertions verify that dimensions match considering optional transposition of the matrix.
        ///
        /// @tparam T Tensor element type.
        /// @param m Input matrix.
        /// @param vmult Vector to multiply (must match matrix column count or row count if transposed).
        /// @param vres Result vector (must match matrix row count or column count if transposed).
        /// @param trans Optional transpose operation on matrix (default: CUBLAS_OP_N for no transpose).
        /// @return Tuple of (nrow, ncol) dimensions of the matrix.
        template <typename T>
        std::tuple<size_t, size_t> check_size_mv(const Matrix_<T> &m, const Tensor_<T> &vmult, const Tensor_<T> &vres, cublasOperation_t trans = CUBLAS_OP_N)
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

        /// @brief Validates tensor shapes for lower triangular matrix-vector multiplication.
        ///
        /// Checks that the lower triangular matrix and vector dimensions are compatible
        /// for matrix-vector product. Assertions verify compatibility considering optional transposition.
        ///
        /// @tparam T Tensor element type.
        /// @param m Input lower triangular matrix.
        /// @param vmult Vector to multiply (must match matrix column count or row count if transposed).
        /// @param vres Result vector (must match matrix row count or column count if transposed).
        /// @param trans Optional transpose operation on matrix (default: CUBLAS_OP_N for no transpose).
        /// @return Tuple of (nrow, ncol) dimensions of the matrix.
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

        /// @brief Calculates leading dimension for column-major matrix storage.
        ///
        /// Computes the leading dimension (row stride) for cuBLAS operations considering
        /// optional matrix transposition for proper column-major memory layout.
        ///
        /// @param nrow Number of rows.
        /// @param ncol Number of columns.
        /// @param transa Optional transpose operation (default: CUBLAS_OP_N for no transpose).
        /// @return Leading dimension value for column-major storage.
        size_t get_leading(size_t nrow, size_t ncol, cublasOperation_t transa = CUBLAS_OP_N);

        /// @brief Converts character transposition flag to cuBLAS operation type.
        ///
        /// Translates string transposition indicators ('N', 'T', 'C') to corresponding
        /// cuBLAS operation types for use in BLAS function calls.
        ///
        /// @param T Transposition character: 'N' (no transpose), 'T' (transpose), 'C' (conjugate transpose).
        /// @return cuBLAS operation type (CUBLAS_OP_N, CUBLAS_OP_T, or CUBLAS_OP_C).
        cublasOperation_t get_trans(const char *T);

        /// @brief Validates tensor shapes for matrix-matrix multiplication operation.
        ///
        /// Checks that matrix dimensions are compatible for C = alpha*A*B + beta*C operation.
        /// Assertions verify dimension compatibility considering optional transposition of A and B.
        ///
        /// @tparam T Element type for matrix A.
        /// @tparam U Element type for matrix B.
        /// @tparam V Element type for matrix C.
        /// @param a First input matrix.
        /// @param b Second input matrix.
        /// @param c Output matrix.
        /// @param transa Optional transpose operation for A (default: CUBLAS_OP_N).
        /// @param transb Optional transpose operation for B (default: CUBLAS_OP_N).
        /// @return Tuple of (m, n, k) dimensions where m=rows of result, n=cols of result, k=reduction dimension.
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

        /// @brief Validates matrix dimensions for matrix-matrix operations with optional transposition.
        ///
        /// Asserts that matrices a, b, and c have compatible dimensions for element-wise operations,
        /// accounting for optional transposition flags. Validates that a and b have matching dimensions
        /// after transposition, and that both match c's dimensions.
        ///
        /// @tparam T Floating-point type.
        /// @param a First input matrix.
        /// @param b Second input matrix.
        /// @param c Output/result matrix.
        /// @param transa Transposition flag for matrix a (CUBLAS_OP_N or CUBLAS_OP_T).
        /// @param transb Transposition flag for matrix b (CUBLAS_OP_N or CUBLAS_OP_T).
        /// @return Tuple (nrowc, ncolc) containing output matrix dimensions.
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

    } // namespace gpu
} // namespace lahva