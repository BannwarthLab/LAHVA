/// @file utils.hpp
/// @brief CPU utility functions for tensor and matrix dimension validation.

#pragma once
#include "linalg.hpp"
#include "const.h"
#include <tuple>

#include <assert.h>

namespace lahva{
    /// @brief Check that two tensors have equal size.
    ///
    /// Asserts that the total number of elements in v1 equals the total number
    /// of elements in v2. This is used to validate tensor arguments in operations
    /// that require operands of compatible dimensions.
    ///
    /// @tparam T Element type of first tensor.
    /// @tparam U Element type of second tensor.
    /// @param v1 First tensor to check.
    /// @param v2 Second tensor to check.
    template<typename T, typename U>
    void check_equal_size(const Tensor<T>& v1, const Tensor<U>& v2){
        assert(v1.size() == v2.size());
    };

    namespace cpu
    {
    
    /// @brief Validate matrix and vector dimensions for matrix-vector multiplication.
    ///
    /// Checks that the matrix and both vector operands have compatible dimensions
    /// for matrix-vector multiplication y := op(A) * x. The validation depends on
    /// whether the matrix is transposed: if not transposed, the matrix column count
    /// must equal x size and row count must equal result vector size. If transposed,
    /// the matrix row count must equal x size and column count must equal result size.
    ///
    /// @tparam T Element type of matrix and vectors.
    /// @param m Matrix operand (Matrix_<T>).
    /// @param vmult Input vector x that is multiplied by the matrix.
    /// @param vres Output/result vector y.
    /// @param trans Transpose flag for the matrix (default: CblasNoTrans).
    /// @return Tuple of (nrow, ncol) - row and column counts of the matrix.
    template<typename MatrixType, typename VectorType>
    std::tuple<BLAS_INT, BLAS_INT> check_size_mv(const MatrixType& m, const VectorType& vmult, const VectorType& vres, CBLAS_TRANSPOSE trans = CblasNoTrans){
        Shape s = m.shape();
        BLAS_INT nrow = s.first;
        BLAS_INT ncol = s.second;
        if (trans == CblasNoTrans) {
            assert(nrow == (BLAS_INT)vres.size());
            assert(ncol == (BLAS_INT)vmult.size());
            
        }
        else {
            assert(nrow == (BLAS_INT)vmult.size());
            assert(ncol == (BLAS_INT)vres.size());
            
        }
        
        return std::make_tuple(nrow, ncol);
    };

    /// @brief Validate triangular matrix and vector dimensions for triangular matrix-vector multiplication.
    ///
    /// Checks that the triangular matrix and both vector operands have compatible dimensions
    /// for triangular matrix-vector multiplication x := op(A) * x. The validation depends on
    /// whether the matrix is transposed: if not transposed, the matrix column count must equal
    /// input vector size and row count must equal result vector size. If transposed, the matrix
    /// row count must equal input vector size and column count must equal result size.
    ///
    /// @tparam T Element type of matrix and vectors.
    /// @param m Triangular matrix operand (LowTriMatrix_<T>) in packed storage format.
    /// @param vmult Input vector that is multiplied by the matrix.
    /// @param vres Output/result vector.
    /// @param trans Transpose flag for the matrix (default: CblasNoTrans).
    /// @return Tuple of (nrow, ncol) - row and column counts of the triangular matrix.
    template<typename T>
    std::tuple<BLAS_INT, BLAS_INT> check_size_mv(const LowTriMatrix_<T>& m, const Tensor<T>& vmult, const Tensor<T>& vres, CBLAS_TRANSPOSE trans = CblasNoTrans){
        Shape s = m.shape();
        BLAS_INT nrow = s.first;
        BLAS_INT ncol = s.second;
        if (trans == CblasNoTrans) {
            assert(nrow == (BLAS_INT)vres.size());
            assert(ncol == (BLAS_INT)vmult.size());
            
        }
        else {
            assert(nrow == (BLAS_INT)vmult.size());
            assert(ncol == (BLAS_INT)vres.size());
            
        }
        
        return std::make_tuple(nrow, ncol);
    };

    /// @brief Validate matrix dimensions for matrix-matrix multiplication.
    ///
    /// Checks that three matrices have compatible dimensions for the operation
    /// C := op(A) * op(B) where op() indicates optional transpose. The validation
    /// ensures that: dimensions of the output matrix C match the untracked transposed
    /// dimensions of A and B, the inner dimension of op(A) matches the outer dimension
    /// of op(B), and all assertions will fail if dimensions are incompatible.
    ///
    /// @tparam T Element type of all matrices.
    /// @param a Left-hand matrix operand (Matrix_<T>).
    /// @param b Right-hand matrix operand (Matrix_<T>).
    /// @param c Output matrix (Matrix_<T>) to store the result.
    /// @param transa Transpose flag for matrix A (default: CblasNoTrans).
    /// @param transb Transpose flag for matrix B (default: CblasNoTrans).
    /// @return Tuple of (m, n, k) where m and n are dimensions of C, and k is the inner dimension.
    template<typename MatrixTypeA, typename MatrixTypeB, typename MatrixTypeC>
    std::tuple<BLAS_INT, BLAS_INT, BLAS_INT> check_size_mm(const MatrixTypeA& a, const MatrixTypeB& b, 
        const MatrixTypeC& c, CBLAS_TRANSPOSE transa = CblasNoTrans, CBLAS_TRANSPOSE transb = CblasNoTrans){
        
        Shape sa = a.shape();
        BLAS_INT nrowa = sa.first;
        BLAS_INT ncola = sa.second;

        Shape sb = b.shape();
        BLAS_INT nrowb = sb.first;
        BLAS_INT ncolb = sb.second;

        Shape sc = c.shape();
        BLAS_INT nrowc = sc.first;
        BLAS_INT ncolc = sc.second;

        BLAS_INT k;

        if (transa == CblasNoTrans) {
            if (transb == CblasNoTrans)
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
           if (transb == CblasNoTrans)
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

    /// @brief Validate that matrices have the same shape for element-wise operations.
    ///
    /// Checks that three matrices have compatible dimensions for element-wise operations
    /// like matrix addition where op(A) and op(B) must have the same dimensions, and C
    /// must have the same dimensions as the results. This is stricter than matrix-matrix
    /// multiplication as it requires element-wise compatibility.
    ///
    /// @tparam T Element type of all matrices.
    /// @param a First input matrix (Matrix_<T>).
    /// @param b Second input matrix (Matrix_<T>).
    /// @param c Output matrix (Matrix_<T>).
    /// @param transa Transpose flag for matrix A (default: CblasNoTrans).
    /// @param transb Transpose flag for matrix B (default: CblasNoTrans).
    /// @return Tuple of (nrow, ncol) - dimensions of the output matrix.
    template<typename MatrixTypeA, typename MatrixTypeB, typename MatrixTypeC>
    std::tuple<BLAS_INT, BLAS_INT> check_same_shape_mm(const MatrixTypeA& a, const MatrixTypeB& b, 
        const MatrixTypeC& c, CBLAS_TRANSPOSE transa = CblasNoTrans, CBLAS_TRANSPOSE transb = CblasNoTrans){
        
        Shape sa = a.shape();
        BLAS_INT nrowa = sa.first;
        BLAS_INT ncola = sa.second;

        Shape sb = b.shape();
        BLAS_INT nrowb = sb.first;
        BLAS_INT ncolb = sb.second;

        Shape sc = c.shape();
        BLAS_INT nrowc = sc.first;
        BLAS_INT ncolc = sc.second;

        if (transa == CblasNoTrans) {
            if (transb == CblasNoTrans)
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
           if (transb == CblasNoTrans)
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

    /// @brief Validate triangular matrix and vector dimensions for triangular matrix-vector solve.
    ///
    /// Checks that a triangular matrix and input vector have compatible dimensions
    /// for triangular matrix-vector solve operations x := op(A)^-1 * x.
    ///
    /// @tparam T Element type of matrix and vector.
    /// @param m Triangular matrix operand (LowTriMatrix_<T>) in packed storage format.
    /// @param vmult Input vector.
    /// @return Tuple of (nrow, ncol) - row and column counts of the triangular matrix.
    template<typename T>
    std::tuple<BLAS_INT, BLAS_INT> check_size_mv(const LowTriMatrix_<T>& m, const Tensor<T>& vmult)
    {
        Shape s = m.shape();
        BLAS_INT nrow = s.first;
        BLAS_INT ncol = s.second;

        assert(ncol == (BLAS_INT)vmult.size());
       
        return std::make_tuple(nrow, ncol);
    };

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
    BLAS_INT get_leading(BLAS_INT nrow, BLAS_INT ncol, CBLAS_TRANSPOSE trans = CblasNoTrans);

    /// @brief Convert character transpose flag to CBLAS_TRANSPOSE enumeration.
    ///
    /// Translates a character representation of transpose operation to the corresponding
    /// CBLAS_TRANSPOSE enum value: "N" -> CblasNoTrans, "T" -> CblasTrans, "C" -> CblasConjTrans.
    ///
    /// @param T Character string "N" (no transpose), "T" (transpose), or "C" (conjugate transpose).
    /// @return Corresponding CBLAS_TRANSPOSE enumeration value.
    CBLAS_TRANSPOSE get_trans(const char* T);

} // namespace cpu
} // namespace lahva