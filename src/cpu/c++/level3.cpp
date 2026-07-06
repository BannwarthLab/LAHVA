/// @file level3.cpp
/// @brief CPU C++ template implementations of Level-3 BLAS operations.
///
/// Provides C++ template functions for Level-3 BLAS matrix-matrix operations including
/// matrix multiplication. Supports optional transposition and scalar factors. Functions
/// accept Matrix<T> types and forward to lower-level pointer-based CBLAS GEMM routines.

#include "../utils/utils.hpp"
#include "impl/blas/cpu/level3.h"
#include "impl/blas/cpu/level3.hpp"
#include "linalg.hpp"

namespace lahva
{
    namespace cpu
    {
        
    /// @brief C++-style wrapper that multiplies two matrices stored in Matrix_<T> and
    ///        forwards the call to the lower-level pointer-based routine.
    ///
    /// This template adapts the C-style MatrixMatrixProduct that takes raw pointers
    /// into a convenient Matrix_<T>-based interface. It supports real and complex
    /// element types (T == double/float/complex_double/complex_float).
    ///
    /// @tparam T Numeric element type of matrices (matches Matrix_<T> template).
    /// @param Ta    Transpose option for A: "N" (no transpose, A), "T" (transpose, A^T), "C" (conjugate-transpose, A^H).
    /// @param Tb    Transpose option for B: "N" (no transpose, B), "T" (transpose, B^T), "C" (conjugate-transpose, B^H).
    /// @param alpha Scaling factor applied to op(A)*op(B).
    /// @param a     Left-hand input matrix (Matrix_<T>).
    /// @param b     Right-hand input matrix (Matrix_<T>).
    /// @param beta  Scaling factor applied to existing contents of c.
    /// @param c     Output matrix (Matrix_<T>) which receives the result.
        template<typename T>
        void MatrixMatrixProduct(const char *Ta, const char *Tb, const T alpha, const Matrix_<T> &a, const Matrix_<T> &b,
                                 const T beta, Matrix_<T> &c)
        {
            CBLAS_TRANSPOSE transa = get_trans(Ta);
            CBLAS_TRANSPOSE transb = get_trans(Tb);
            int m, n, k;
            std::tie(m, n, k) = check_size_mm(a, b, c, transa, transb);
            MatrixMatrixProduct(Ta, Tb, m, n, k, alpha, a.data(), b.data(), beta, c.data());
        };

        template void MatrixMatrixProduct(const char *Ta, const char *Tb, const double alpha, const Matrix_<double> &a, const Matrix_<double> &b,
                                 const double beta, Matrix_<double> &c);
        template void MatrixMatrixProduct(const char *Ta, const char *Tb, const float alpha, const Matrix_<float> &a, const Matrix_<float> &b,
                                 const float beta, Matrix_<float> &c); 
        template void MatrixMatrixProduct(const char *Ta, const char *Tb, const complex_double alpha, const Matrix_<complex_double> &a, const Matrix_<complex_double> &b,
                                 const complex_double beta, Matrix_<complex_double> &c);
        template void MatrixMatrixProduct(const char *Ta, const char *Tb, const complex_float alpha, const Matrix_<complex_float> &a, const Matrix_<complex_float> &b,
                                 const complex_float beta, Matrix_<complex_float> &c);


        /// @brief Block-diagonal matrix-matrix multiply with full transpose support (double precision).
        ///
        /// Performs C = alpha * op(A) * op(B) + beta * C where A is a block-diagonal matrix, B and C are dense matrices,
        /// and op(X) is X, X^T, or X^H depending on transpose flags `Ta` and `Tb`. Uses MKL batch GEMM to efficiently
        /// process groups of blocks with identical dimensions, falling back to sequential processing if batch GEMM is unavailable.
        ///
        /// @param Ta    Transpose option for A: "N" (no transpose, A), "T" (transpose, A^T), "C" (conjugate-transpose, A^H).
        /// @param Tb    Transpose option for B: "N" (no transpose, B), "T" (transpose, B^T), "C" (conjugate-transpose, B^H).
        /// @param alpha Scaling factor applied to op(A)*op(B).
        /// @param a     Block-diagonal input matrix (BlockDiagMatrix_<double>).
        /// @param b     Dense input matrix (Matrix_<double>).
        /// @param beta  Scaling factor applied to existing contents of c.
        /// @param c     Output dense matrix (Matrix_<double>).
        template <>
        void MatrixMatrixProduct(const char *Ta, const char *Tb, const double alpha, const BlockDiagMatrix_<double> &a, const Matrix_<double> &b,
                                 const double beta, Matrix_<double> &c)
        {

            CBLAS_TRANSPOSE transa = get_trans(Ta);
            CBLAS_TRANSPOSE transb = get_trans(Tb);

            auto b_shape = b.shape();
            int b_rows = static_cast<int>(b_shape.first);
            int b_cols = static_cast<int>(b_shape.second);

            int m, n, k;
            std::tie(m, n, k) = check_size_mm(a, b, c, transa, transb);

            std::vector<CBLAS_TRANSPOSE> transa_array, transb_array;
            std::vector<double> alpha_array, beta_array;
            std::vector<MKL_INT> n_array, ldb_array, ldc_array, m_array, k_array, lda_array, group_size;
            std::vector<const double*> a_array, b_array;
            std::vector<double*> c_array;

            size_t i = 0;
            while (i < a.num_blocks()) {
                Shape block_shape = a.get_block_shape(i);
                MKL_INT block_m = static_cast<MKL_INT>(block_shape.first);
                MKL_INT block_k = static_cast<MKL_INT>(block_shape.second);
                
                // Use transposed dimensions for grouping if transpose is applied
                MKL_INT current_m = (transa == CblasTrans) ? block_k : block_m;
                MKL_INT current_k = (transa == CblasTrans) ? block_m : block_k;
                size_t group_start = i;

                // Find consecutive blocks with same dimensions (accounting for transpose)
                while (i < a.num_blocks()) {
                    Shape shape = a.get_block_shape(i);
                    MKL_INT shape_m = static_cast<MKL_INT>(shape.first);
                    MKL_INT shape_k = static_cast<MKL_INT>(shape.second);
                    MKL_INT logical_m = (transa == CblasTrans) ? shape_k : shape_m;
                    MKL_INT logical_k = (transa == CblasTrans) ? shape_m : shape_k;
                    if (logical_m == current_m && logical_k == current_k) {
                        ++i;
                    } else {
                        break;
                    }
                }

                size_t group_count = i - group_start;

                // Add this group to the batch arrays
                transa_array.push_back(transa);
                transb_array.push_back(transb);
                alpha_array.push_back(alpha);
                beta_array.push_back(beta);
                n_array.push_back(n);
                ldb_array.push_back(b_rows);
                ldc_array.push_back(m);
                m_array.push_back(current_m);
                k_array.push_back(current_k);
                lda_array.push_back(block_m);
                group_size.push_back(group_count);

                // Add pointers for all blocks in this group
                for (size_t j = group_start; j < i; ++j) {
                    a_array.push_back(static_cast<const double*>(a.get_block_data(j)));
                    b_array.push_back(b.data() + a.get_col_offsets()[j]);
                    c_array.push_back(c.data() + a.get_row_offsets()[j]);
                }
            }

            MKL_INT group_count = group_size.size();

            cblas_dgemm_batch(
                major,
                transa_array.data(),
                transb_array.data(),
                m_array.data(),
                n_array.data(),
                k_array.data(),
                alpha_array.data(),
                a_array.data(),
                lda_array.data(),
                b_array.data(),
                ldb_array.data(),
                beta_array.data(),
                c_array.data(),
                ldc_array.data(),
                group_count,
                group_size.data()
            );
        }

        /// @brief Block-diagonal matrix-matrix multiply with full transpose support (single precision).
        ///
        /// Performs C = alpha * op(A) * op(B) + beta * C where A is a block-diagonal matrix, B and C are dense matrices,
        /// and op(X) is X, X^T, or X^H depending on transpose flags `Ta` and `Tb`. Uses MKL batch GEMM to efficiently
        /// process groups of blocks with identical dimensions, falling back to sequential processing if batch GEMM is unavailable.
        /// Float variant.
        ///
        /// @param Ta    Transpose option for A: "N" (no transpose, A), "T" (transpose, A^T), "C" (conjugate-transpose, A^H).
        /// @param Tb    Transpose option for B: "N" (no transpose, B), "T" (transpose, B^T), "C" (conjugate-transpose, B^H).
        /// @param alpha Scaling factor applied to op(A)*op(B).
        /// @param a     Block-diagonal input matrix (BlockDiagMatrix_<float>).
        /// @param b     Dense input matrix (Matrix_<float>).
        /// @param beta  Scaling factor applied to existing contents of c.
        /// @param c     Output dense matrix (Matrix_<float>).
        template <>
        void MatrixMatrixProduct(const char *Ta, const char *Tb, const float alpha, const BlockDiagMatrix_<float> &a, const Matrix_<float> &b,
                                 const float beta, Matrix_<float> &c)
        {

            CBLAS_TRANSPOSE transa = get_trans(Ta);
            CBLAS_TRANSPOSE transb = get_trans(Tb);

            auto b_shape = b.shape();
            int b_rows = static_cast<int>(b_shape.first);
            int b_cols = static_cast<int>(b_shape.second);

            int m, n, k;
            std::tie(m, n, k) = check_size_mm(a, b, c, transa, transb);


            std::vector<CBLAS_TRANSPOSE> transa_array, transb_array;
            std::vector<float> alpha_array, beta_array;
            std::vector<MKL_INT> n_array, ldb_array, ldc_array, m_array, k_array, lda_array, group_size;
            std::vector<const float*> a_array, b_array;
            std::vector<float*> c_array;

            size_t i = 0;
            while (i < a.num_blocks()) {
                Shape block_shape = a.get_block_shape(i);
                MKL_INT block_m = static_cast<MKL_INT>(block_shape.first);
                MKL_INT block_k = static_cast<MKL_INT>(block_shape.second);
                
                // Use transposed dimensions for grouping if transpose is applied
                MKL_INT current_m = (transa == CblasTrans) ? block_k : block_m;
                MKL_INT current_k = (transa == CblasTrans) ? block_m : block_k;
                size_t group_start = i;

                // Find consecutive blocks with same dimensions (accounting for transpose)
                while (i < a.num_blocks()) {
                    Shape shape = a.get_block_shape(i);
                    MKL_INT shape_m = static_cast<MKL_INT>(shape.first);
                    MKL_INT shape_k = static_cast<MKL_INT>(shape.second);
                    MKL_INT logical_m = (transa == CblasTrans) ? shape_k : shape_m;
                    MKL_INT logical_k = (transa == CblasTrans) ? shape_m : shape_k;
                    if (logical_m == current_m && logical_k == current_k) {
                        ++i;
                    } else {
                        break;
                    }
                }

                size_t group_count = i - group_start;

                transa_array.push_back(transa);
                transb_array.push_back(transb);
                alpha_array.push_back(alpha);
                beta_array.push_back(beta);
                n_array.push_back(n);
                ldb_array.push_back(b_rows);
                ldc_array.push_back(m);
                m_array.push_back(current_m);
                k_array.push_back(current_k);
                lda_array.push_back(block_m);
                group_size.push_back(group_count);

                // Add pointers for all blocks in this group
                for (size_t j = group_start; j < i; ++j) {
                    a_array.push_back(static_cast<const float*>(a.get_block_data(j)));
                    b_array.push_back(b.data() + a.get_col_offsets()[j]);
                    c_array.push_back(c.data() + a.get_row_offsets()[j]);
                }
            }

            MKL_INT group_count = group_size.size();

            cblas_sgemm_batch(
                major,
                transa_array.data(),
                transb_array.data(),
                m_array.data(),
                n_array.data(),
                k_array.data(),
                alpha_array.data(),
                a_array.data(),
                lda_array.data(),
                b_array.data(),
                ldb_array.data(),
                beta_array.data(),
                c_array.data(),
                ldc_array.data(),
                group_count,
                group_size.data()
            );
        }

        /// @brief Dense matrix times block-diagonal matrix with full transpose support (double precision).
        ///
        /// Performs C = alpha * op(A) * op(B) + beta * C where B is a block-diagonal matrix, A and C are dense matrices,
        /// and op(X) is X, X^T, or X^H depending on transpose flags `Ta` and `Tb`. Uses MKL batch GEMM to efficiently
        /// process groups of blocks with identical dimensions, falling back to sequential processing if batch GEMM is unavailable.
        ///
        /// @param Ta    Transpose option for A: "N" (no transpose, A), "T" (transpose, A^T), "C" (conjugate-transpose, A^H).
        /// @param Tb    Transpose option for B: "N" (no transpose, B), "T" (transpose, B^T), "C" (conjugate-transpose, B^H).
        /// @param alpha Scaling factor applied to op(A)*op(B).
        /// @param a     Dense input matrix (Matrix_<double>).
        /// @param b     Block-diagonal input matrix (BlockDiagMatrix_<double>).
        /// @param beta  Scaling factor applied to existing contents of c.
        /// @param c     Output dense matrix (Matrix_<double>).
        template <>
        void MatrixMatrixProduct(const char *Ta, const char *Tb, const double alpha, const Matrix_<double> &a, const BlockDiagMatrix_<double> &b,
                                 const double beta, Matrix_<double> &c)
        {

            CBLAS_TRANSPOSE transa = get_trans(Ta);
            CBLAS_TRANSPOSE transb = get_trans(Tb);

            auto a_shape = a.shape();
            int m = static_cast<int>(a_shape.first);
            int k = static_cast<int>(a_shape.second);

            int m_out, n_out, k_out;
            std::tie(m_out, n_out, k_out) = check_size_mm(a, b, c, transa, transb);

            std::vector<CBLAS_TRANSPOSE> transa_array, transb_array;
            std::vector<double> alpha_array, beta_array;
            std::vector<MKL_INT> n_array, ldb_array, ldc_array, m_array, k_array, lda_array, group_size;
            std::vector<const double*> a_array, b_array;
            std::vector<double*> c_array;

            size_t i = 0;
            while (i < b.num_blocks()) {
                Shape block_shape = b.get_block_shape(i);
                MKL_INT block_k = static_cast<MKL_INT>(block_shape.first);
                MKL_INT block_n = static_cast<MKL_INT>(block_shape.second);
                
                // Use transposed dimensions for grouping if transpose is applied
                MKL_INT current_k = (transb == CblasTrans) ? block_n : block_k;
                MKL_INT current_n = (transb == CblasTrans) ? block_k : block_n;
                size_t group_start = i;

                // Find consecutive blocks with same dimensions (accounting for transpose)
                while (i < b.num_blocks()) {
                    Shape shape = b.get_block_shape(i);
                    MKL_INT shape_k = static_cast<MKL_INT>(shape.first);
                    MKL_INT shape_n = static_cast<MKL_INT>(shape.second);
                    MKL_INT logical_k = (transb == CblasTrans) ? shape_n : shape_k;
                    MKL_INT logical_n = (transb == CblasTrans) ? shape_k : shape_n;
                    if (logical_k == current_k && logical_n == current_n) {
                        ++i;
                    } else {
                        break;
                    }
                }

                size_t group_count = i - group_start;

                // Determine m_val based on A's transpose flag
                MKL_INT m_val = (transa == CblasTrans) ? k : m;

                // Add this group to the batch arrays
                transa_array.push_back(transa);
                transb_array.push_back(transb);
                alpha_array.push_back(alpha);
                beta_array.push_back(beta);
                n_array.push_back(current_n);
                ldb_array.push_back(block_k);
                ldc_array.push_back(m_out);
                m_array.push_back(m_val);
                k_array.push_back(current_k);
                lda_array.push_back(m);
                group_size.push_back(group_count);

                // Add pointers for all blocks in this group
                for (size_t j = group_start; j < i; ++j) {
                    a_array.push_back(a.data() + b.get_row_offsets()[j] * m);
                    b_array.push_back(static_cast<const double*>(b.get_block_data(j)));
                    c_array.push_back(c.data() + b.get_col_offsets()[j] * m);
                }
            }

            MKL_INT group_count = group_size.size();

            cblas_dgemm_batch(
                major,
                transa_array.data(),
                transb_array.data(),
                m_array.data(),
                n_array.data(),
                k_array.data(),
                alpha_array.data(),
                a_array.data(),
                lda_array.data(),
                b_array.data(),
                ldb_array.data(),
                beta_array.data(),
                c_array.data(),
                ldc_array.data(),
                group_count,
                group_size.data()
            );
        }

        /// @brief Dense matrix times block-diagonal matrix with full transpose support (single precision).
        ///
        /// Performs C = alpha * op(A) * op(B) + beta * C where B is a block-diagonal matrix, A and C are dense matrices,
        /// and op(X) is X, X^T, or X^H depending on transpose flags `Ta` and `Tb`. Uses MKL batch GEMM to efficiently
        /// process groups of blocks with identical dimensions, falling back to sequential processing if batch GEMM is unavailable.
        /// Float variant.
        ///
        /// @param Ta    Transpose option for A: "N" (no transpose, A), "T" (transpose, A^T), "C" (conjugate-transpose, A^H).
        /// @param Tb    Transpose option for B: "N" (no transpose, B), "T" (transpose, B^T), "C" (conjugate-transpose, B^H).
        /// @param alpha Scaling factor applied to op(A)*op(B).
        /// @param a     Dense input matrix (Matrix_<float>).
        /// @param b     Block-diagonal input matrix (BlockDiagMatrix_<float>).
        /// @param beta  Scaling factor applied to existing contents of c.
        /// @param c     Output dense matrix (Matrix_<float>).
        template <>
        void MatrixMatrixProduct(const char *Ta, const char *Tb, const float alpha, const Matrix_<float> &a, const BlockDiagMatrix_<float> &b,
                                 const float beta, Matrix_<float> &c)
        {

            CBLAS_TRANSPOSE transa = get_trans(Ta);
            CBLAS_TRANSPOSE transb = get_trans(Tb);

            auto a_shape = a.shape();
            int m = static_cast<int>(a_shape.first);
            int k = static_cast<int>(a_shape.second);

            int m_out, n_out, k_out;
            std::tie(m_out, n_out, k_out) = check_size_mm(a, b, c, transa, transb);

            std::vector<CBLAS_TRANSPOSE> transa_array, transb_array;
            std::vector<float> alpha_array, beta_array;
            std::vector<MKL_INT> n_array, ldb_array, ldc_array, m_array, k_array, lda_array, group_size;
            std::vector<const float*> a_array, b_array;
            std::vector<float*> c_array;

            size_t i = 0;
            while (i < b.num_blocks()) {
                Shape block_shape = b.get_block_shape(i);
                MKL_INT block_k = static_cast<MKL_INT>(block_shape.first);
                MKL_INT block_n = static_cast<MKL_INT>(block_shape.second);
                
                // Use transposed dimensions for grouping if transpose is applied
                MKL_INT current_k = (transb == CblasTrans) ? block_n : block_k;
                MKL_INT current_n = (transb == CblasTrans) ? block_k : block_n;
                size_t group_start = i;

                // Find consecutive blocks with same dimensions (accounting for transpose)
                while (i < b.num_blocks()) {
                    Shape shape = b.get_block_shape(i);
                    MKL_INT shape_k = static_cast<MKL_INT>(shape.first);
                    MKL_INT shape_n = static_cast<MKL_INT>(shape.second);
                    MKL_INT logical_k = (transb == CblasTrans) ? shape_n : shape_k;
                    MKL_INT logical_n = (transb == CblasTrans) ? shape_k : shape_n;
                    if (logical_k == current_k && logical_n == current_n) {
                        ++i;
                    } else {
                        break;
                    }
                }

                size_t group_count = i - group_start;

                // Determine m_val based on A's transpose flag
                MKL_INT m_val = (transa == CblasTrans) ? k : m;

                // Add this group to the batch arrays
                transa_array.push_back(transa);
                transb_array.push_back(transb);
                alpha_array.push_back(alpha);
                beta_array.push_back(beta);
                n_array.push_back(current_n);
                ldb_array.push_back(block_k);
                ldc_array.push_back(m_out);
                m_array.push_back(m_val);
                k_array.push_back(current_k);
                lda_array.push_back(m);
                group_size.push_back(group_count);

                // Add pointers for all blocks in this group
                for (size_t j = group_start; j < i; ++j) {
                    a_array.push_back(a.data() + b.get_row_offsets()[j] * m);
                    b_array.push_back(static_cast<const float*>(b.get_block_data(j)));
                    c_array.push_back(c.data() + b.get_col_offsets()[j] * m);
                }
            }

            MKL_INT group_count = group_size.size();

            cblas_sgemm_batch(
                major,
                transa_array.data(),
                transb_array.data(),
                m_array.data(),
                n_array.data(),
                k_array.data(),
                alpha_array.data(),
                a_array.data(),
                lda_array.data(),
                b_array.data(),
                ldb_array.data(),
                beta_array.data(),
                c_array.data(),
                ldc_array.data(),
                group_count,
                group_size.data()
            );
        }

    /// @brief Symmetric matrix-matrix multiply (double precision).
    ///
    /// Performs C = alpha * A * B + beta * C when `side == CblasLeft`, or
    /// C = alpha * B * A + beta * C when `side == CblasRight`. Matrix A is
    /// assumed to be symmetric and stored according to the project's triangular
    /// convention (see `tri` in `const.h`). This wrapper adapts Matrix_<double>
    /// to the cblas_dsymm call.
    ///
    /// @param side  Side on which the symmetric matrix A appears (CblasLeft/CblasRight).
    /// @param alpha Scaling factor for the product involving A.
    /// @param a     Symmetric matrix A (Matrix_<double>).
    /// @param b     Matrix B (Matrix_<double>).
    /// @param beta  Scaling factor for existing contents of C.
    /// @param c     Output matrix C (Matrix_<double>), receives the result.
        void SymMatrixMatrixProduct(const CBLAS_SIDE side, const double alpha, const Matrix_<double> &a, const Matrix_<double> &b,
                                    const double beta, Matrix_<double> &c)
        {
            int m, n, k;
            std::tie(m, n, k) = check_size_mm(a, b, c);
            BLAS_INT lda = get_leading(m, k);
            BLAS_INT ldb = get_leading(k, n);
            BLAS_INT ldc = get_leading(m, n);
            cblas_dsymm(major, side, tri, m, n, alpha, a.data(), lda, b.data(), ldb, beta, c.data(), ldc);
        };

    /// @brief Symmetric matrix-matrix multiply (single precision).
    ///
    /// Float variant of SymMatrixMatrixProduct. Performs the same operation
    /// as the double-precision variant but calls cblas_ssymm under the hood.
    ///
    /// @param side  Side on which the symmetric matrix A appears (CblasLeft/CblasRight).
    /// @param alpha Scaling factor for the product involving A (float).
    /// @param a     Symmetric matrix A (Matrix_<float>).
    /// @param b     Matrix B (Matrix_<float>).
    /// @param beta  Scaling factor for existing contents of C (float).
    /// @param c     Output matrix C (Matrix_<float>), receives the result.
        void SymMatrixMatrixProduct(const CBLAS_SIDE side, const float alpha, const Matrix_<float> &a, const Matrix_<float> &b,
                                    const float beta, Matrix_<float> &c)
        {
            int m, n, k;
            std::tie(m, n, k) = check_size_mm(a, b, c);

            BLAS_INT lda = get_leading(m, k);
            BLAS_INT ldb = get_leading(k, n);
            BLAS_INT ldc = get_leading(m, n);

            cblas_ssymm(major, side, tri, m, n, alpha, a.data(), lda, b.data(), ldb, beta, c.data(), ldc);
        };

    } // namespace cpu
} // namespace lahva
