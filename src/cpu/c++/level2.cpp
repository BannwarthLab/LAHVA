#include "linalg.hpp"
#include "impl/blas/cpu/level2.hpp"
#include "../utils/utils.hpp"
#include "impl/blas/cpu/level2.h"

namespace lahva
{
    namespace cpu
    {
        /// @brief Simple interface to DGER \f$\mathbf{A}=alpha*\vec{x}*\vec{y}**T + A\f$
        /// @param x Vector x
        /// @param incx stride of Vector x
        /// @param y vector y
        /// @param incy stride of Vector y
        /// @param alpha value by which \f$\vec{x}*\vec{y}**T\f$ is scaled
        /// @param A matrix A of size ndimX by ndimY
        void OuterVectorProduct(const Vector<double>& x, const Vector<double>& y, Matrix<double>& A, size_t incx, size_t incy, const double alpha) {
            
            if (A.shape().first != x.size() || A.shape().second != y.size()) {
                throw std::invalid_argument("OuterVectorProduct: Output matrix A has incorrect shape.");
            }

            OuterVectorProduct(x.size(), x.data(), incx, y.size(), y.data(), incy, alpha, A.data());
        }

        /// @brief Simple interface to SGER \f$\mathbf{A}=alpha*\vec{x}*\vec{y}**T + A\f$
        /// @param x Vector x
        /// @param incx stride of Vector x
        /// @param y vector y
        /// @param incy stride of Vector y
        /// @param alpha value by which \f$\vec{x}*\vec{y}**T\f$ is scaled
        /// @param A matrix A of size ndimX by ndimY
        void OuterVectorProduct(const Vector<float>& x, const Vector<float>& y, Matrix<float>& A, size_t incx, size_t incy, const float alpha) {
            if (A.shape().first != x.size() || A.shape().second != y.size()) {
                throw std::invalid_argument("OuterVectorProduct: Output matrix A has incorrect shape.");
            }

            OuterVectorProduct(x.size(), x.data(), incx, y.size(), y.data(), incy, alpha, A.data());
        }
       
       
        /*! @brief Simple interface to DGEMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        or \f$\vec{y}=alpha*\mathbf{A}^\intercal*\vec{x}+beta*\vec{y}\f$ or
        \f$\vec{y}=alpha*conj(\mathbf{A}^\intercal)*\vec{x}+beta*\vec{y}\f$  for specified stride
            @param[in] T character representing the equation execute "N", "T", or "C"
            @param[in] alpha double value by which A*x is scaled
            @param[in] a pointer to the A matrix in column-major ordering
            @param[in] x pointer to the Vector_ x
            @param[in] incx stride of Vector_ x
            @param[in] beta double value by which y is scaled
            @param[in,out] y pointer to the y Vector_ values
            @param[in] incy stride of Vector_ y
        */
        void MatrixVectorProduct(const char *T, const double alpha, const Matrix_<double> &a,
                                 const Vector_<double> &x, const size_t incx, const double beta, Vector_<double> &y, const size_t incy)
        {
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;
            CBLAS_TRANSPOSE trans = get_trans(T);
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y, trans);
            BLAS_INT lda = get_leading(nrow, ncol);

            cblas_dgemv(major, trans, nrow, ncol, alpha, a.data(), lda, x.data(), inx, beta, y.data(), iny);
        };

        /*! @brief Simple interface to DGEMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
       or \f$\vec{y}=alpha*\mathbf{A}^\intercal*\vec{x}+beta*\vec{y}\f$ or
       \f$\vec{y}=alpha*conj(\mathbf{A}^\intercal)*\vec{x}+beta*\vec{y}\f$ for specified stride
           @param[in] a pointer to the A matrix in column-major ordering
           @param[in] x pointer to the Vector_ x
           @param[in,out] y pointer to the y Vector_ values
           @param[in] T (optional, default value "N") character representing the equation execute "N", "T", or "C"
           @param[in] alpha (optional, default 1.0) double value by which A*x is scaled
           @param[in] beta (optional, default 0.0) double value by which y is scaled
           @param[in] incx (optional, default 1) stride of Vector_ x
           @param[in] incy (optional, default 1) stride of Vector_ y
       */
        void MatrixVectorProduct(const Matrix_<double> &a, const Vector_<double> &x, Vector_<double> &y, const char *T,
                                 const double alpha, const double beta, const size_t incx, const size_t incy)
        {
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;
            CBLAS_TRANSPOSE trans = get_trans(T);
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y, trans);
            BLAS_INT lda = get_leading(nrow, ncol);
            cblas_dgemv(major, trans, nrow, ncol, alpha, a.data(), lda, x.data(), inx, beta, y.data(), iny);
        };

        /*! @brief Simple interface to DGEMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        or \f$\vec{y}=alpha*\mathbf{A}^\intercal*\vec{x}+beta*\vec{y}\f$ or
        \f$\vec{y}=alpha*conj(\mathbf{A}^\intercal)*\vec{x}+beta*\vec{y}\f$  for specified stride
            @param[in] T character representing the equation execute "N", "T", or "C"
            @param[in] alpha double value by which A*x is scaled
            @param[in] a pointer to the A matrix in column-major ordering
            @param[in] x pointer to the Vector_ x
            @param[in] incx stride of Vector_ x
            @param[in] beta double value by which y is scaled
            @param[in,out] y pointer to the y Vector_ values
            @param[in] incy stride of Vector_ y
        */
        void MatrixVectorProduct(const char *T, const float alpha, const Matrix_<float> &a,
                                 const Vector_<float> &x, const size_t incx, const float beta, Vector_<float> &y, const size_t incy)
        {
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;
            CBLAS_TRANSPOSE trans = get_trans(T);
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y, trans);
            BLAS_INT lda = get_leading(nrow, ncol);
            cblas_sgemv(major, trans, nrow, ncol, alpha, a.data(), lda, x.data(), inx, beta, y.data(), iny);
        };

        /*! @brief Simple interface to SGEMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
       or \f$\vec{y}=alpha*\mathbf{A}^\intercal*\vec{x}+beta*\vec{y}\f$ or
       \f$\vec{y}=alpha*conj(\mathbf{A}^\intercal)*\vec{x}+beta*\vec{y}\f$ for specified stride
           @param[in] a pointer to the A matrix in column-major ordering
           @param[in] x pointer to the Vector_ x
           @param[in,out] y pointer to the y Vector_ values
           @param[in] T (optional, default value "N") character representing the equation execute "N", "T", or "C"
           @param[in] alpha (optional, default 1.0) double value by which A*x is scaled
           @param[in] beta (optional, default 0.0) double value by which y is scaled
           @param[in] incx (optional, default 1) stride of Vector_ x
           @param[in] incy (optional, default 1) stride of Vector_ y
       */
        void MatrixVectorProduct(const Matrix_<float> &a, const Vector_<float> &x, Vector_<float> &y, const char *T,
                                 const float alpha, const float beta, const size_t incx, const size_t incy)
        {
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;
            CBLAS_TRANSPOSE trans = get_trans(T);
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y, trans);
            BLAS_INT lda = get_leading(nrow, ncol);
            cblas_sgemv(major, trans, nrow, ncol, alpha, a.data(), lda, x.data(), inx, beta, y.data(), iny);
        };

        /*! @brief Simple interface to DGEMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        or \f$\vec{y}=alpha*\mathbf{A}^\intercal*\vec{x}+beta*\vec{y}\f$ or
        \f$\vec{y}=alpha*conj(\mathbf{A}^\intercal)*\vec{x}+beta*\vec{y}\f$  for specified stride
            @param[in] T character representing the equation execute "N", "T", or "C"
            @param[in] alpha double value by which A*x is scaled
            @param[in] a pointer to the A matrix in column-major ordering
            @param[in] x pointer to the Vector_ x
            @param[in] incx stride of Vector_ x
            @param[in] beta double value by which y is scaled
            @param[in,out] y pointer to the y Vector_ values
            @param[in] incy stride of Vector_ y
        */
        void MatrixVectorProduct(const char *T, const complex_double alpha, const Matrix_<complex_double> &a, const Vector_<complex_double> &x, 
                                 const size_t incx, const complex_double beta, Vector_<complex_double> &y, const size_t incy)
        {
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;
            CBLAS_TRANSPOSE trans = get_trans(T);
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y, trans);
            BLAS_INT lda = get_leading(nrow, ncol);

            cblas_zgemv(major, trans, nrow, ncol, &alpha, a.data(), lda, x.data(), inx, &beta, y.data(), iny);
        };

        /*! @brief Simple interface to DGEMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        or \f$\vec{y}=alpha*\mathbf{A}^\intercal*\vec{x}+beta*\vec{y}\f$ or
        \f$\vec{y}=alpha*conj(\mathbf{A}^\intercal)*\vec{x}+beta*\vec{y}\f$ for specified stride
        @param[in] a pointer to the A matrix in column-major ordering
        @param[in] x pointer to the Vector_ x
        @param[in,out] y pointer to the y Vector_ values
        @param[in] T (optional, default value "N") character representing the equation execute "N", "T", or "C"
        @param[in] alpha (optional, default 1.0) double value by which A*x is scaled
        @param[in] beta (optional, default 0.0) double value by which y is scaled
        @param[in] incx (optional, default 1) stride of Vector_ x
        @param[in] incy (optional, default 1) stride of Vector_ y
        */
        void MatrixVectorProduct(const Matrix_<complex_double> &a, const Vector_<complex_double> &x, Vector_<complex_double> &y, const char *T,
                                 const complex_double alpha, const complex_double beta, const size_t incx, const size_t incy)
        {
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;
            CBLAS_TRANSPOSE trans = get_trans(T);
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y, trans);
            BLAS_INT lda = get_leading(nrow, ncol);
            cblas_zgemv(major, trans, nrow, ncol, &alpha, a.data(), lda, x.data(), inx, &beta, y.data(), iny);
        };

        /*! @brief Simple interface to DGEMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        or \f$\vec{y}=alpha*\mathbf{A}^\intercal*\vec{x}+beta*\vec{y}\f$ or
        \f$\vec{y}=alpha*conj(\mathbf{A}^\intercal)*\vec{x}+beta*\vec{y}\f$  for specified stride
        @param[in] T character representing the equation execute "N", "T", or "C"
        @param[in] alpha double value by which A*x is scaled
        @param[in] a pointer to the A matrix in column-major ordering
        @param[in] x pointer to the Vector_ x
        @param[in] incx stride of Vector_ x
        @param[in] beta double value by which y is scaled
        @param[in,out] y pointer to the y Vector_ values
        @param[in] incy stride of Vector_ y
        */
        void MatrixVectorProduct(const char *T, const complex_float alpha, const Matrix_<complex_float> &a, const Vector_<complex_float> &x, 
                                 const size_t incx, const complex_float beta, Vector_<complex_float> &y, const size_t incy)
        {
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;
            CBLAS_TRANSPOSE trans = get_trans(T);
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y, trans);
            BLAS_INT lda = get_leading(nrow, ncol);
            cblas_cgemv(major, trans, nrow, ncol, &alpha, a.data(), lda, x.data(), inx, &beta, y.data(), iny);
        };

        /*! @brief Simple interface to SGEMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        or \f$\vec{y}=alpha*\mathbf{A}^\intercal*\vec{x}+beta*\vec{y}\f$ or
        \f$\vec{y}=alpha*conj(\mathbf{A}^\intercal)*\vec{x}+beta*\vec{y}\f$ for specified stride
        @param[in] a pointer to the A matrix in column-major ordering
        @param[in] x pointer to the Vector_ x
        @param[in,out] y pointer to the y Vector_ values
        @param[in] T (optional, default value "N") character representing the equation execute "N", "T", or "C"
        @param[in] alpha (optional, default 1.0) double value by which A*x is scaled
        @param[in] beta (optional, default 0.0) double value by which y is scaled
        @param[in] incx (optional, default 1) stride of Vector_ x
        @param[in] incy (optional, default 1) stride of Vector_ y
        */
        void MatrixVectorProduct(const Matrix_<complex_float> &a, const Vector_<complex_float> &x, Vector_<complex_float> &y, const char *T,
                                 const complex_float alpha, const complex_float beta, const size_t incx, const size_t incy)
        {
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;
            CBLAS_TRANSPOSE trans = get_trans(T);
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y, trans);
            BLAS_INT lda = get_leading(nrow, ncol);
            cblas_cgemv(major, trans, nrow, ncol, &alpha, a.data(), lda, x.data(), inx, &beta, y.data(), iny);
        };

        /// @brief Block-diagonal matrix-vector multiply with full transpose support (double precision).
        ///
        /// Performs y := alpha * op(A) * x + beta * y where A is a block-diagonal matrix and op(A) is A, A^T, or A^H
        /// depending on the transpose flag `Ta`. Each diagonal block is multiplied independently with corresponding
        /// portions of the input vector using BLAS dgemv, then accumulated with optional scaling. Results are
        /// parallelized across blocks using OpenMP.
        ///
        /// @param Ta    Transpose option for A: "N" (no transpose, A), "T" (transpose, A^T), "C" (conjugate-transpose, A^H).
        /// @param alpha Scaling factor applied to op(A)*x.
        /// @param a     Block-diagonal input matrix (BlockDiagMatrix_<double>).
        /// @param x     Input vector (Vector_<double>).
        /// @param incx  Stride between consecutive elements in vector x.
        /// @param beta  Scaling factor applied to existing contents of y.
        /// @param y     Output vector (destination / input-output, Vector_<double>).
        /// @param incy  Stride between consecutive elements in vector y.
        /// @note Benchmarking showed this method to be faster than MKL batched GEMV.
        template <>
        void MatrixVectorProduct(const char *Ta, const double alpha, const BlockDiagMatrix_<double> &a, const Vector_<double> &x,
                                 const size_t incx, const double beta, Vector_<double> &y, const size_t incy)
        {
            CBLAS_TRANSPOSE trans = get_trans(Ta);
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y, trans);

            #pragma omp parallel for schedule(guided)
            for (size_t i = 0; i < a.num_blocks(); ++i) {
                Shape block_shape = a.get_block_shape(i);
                BLAS_INT m = static_cast<BLAS_INT>(block_shape.first);
                BLAS_INT n = static_cast<BLAS_INT>(block_shape.second);

                const double* Ai = static_cast<const double*>(a.get_block_data(i));
                const double* xi = x.data() + a.get_col_offsets()[i];
                double* yi = y.data() + a.get_row_offsets()[i];

                cblas_dgemv(major, trans, m, n, alpha, Ai, m, xi, inx, beta, yi, iny);
            }
        }

        /// @brief Block-diagonal matrix-vector multiply with full transpose support (single precision).
        ///
        /// Performs y := alpha * op(A) * x + beta * y where A is a block-diagonal matrix and op(A) is A, A^T, or A^H
        /// depending on the transpose flag `Ta`. Each diagonal block is multiplied independently with corresponding
        /// portions of the input vector using BLAS sgemv, then accumulated with optional scaling. Results are
        /// parallelized across blocks using OpenMP. Float variant.
        ///
        /// @param Ta    Transpose option for A: "N" (no transpose, A), "T" (transpose, A^T), "C" (conjugate-transpose, A^H).
        /// @param alpha Scaling factor applied to op(A)*x.
        /// @param a     Block-diagonal input matrix (BlockDiagMatrix_<float>).
        /// @param x     Input vector (Vector_<float>).
        /// @param incx  Stride between consecutive elements in vector x.
        /// @param beta  Scaling factor applied to existing contents of y.
        /// @param y     Output vector (destination / input-output, Vector_<float>).
        /// @param incy  Stride between consecutive elements in vector y.
        /// @note Benchmarking showed this method to be faster than MKL batched GEMV.
        template <>
        void MatrixVectorProduct(const char *Ta, const float alpha, const BlockDiagMatrix_<float> &a, const Vector_<float> &x,
                                 const size_t incx, const float beta, Vector_<float> &y, const size_t incy)
        {
            CBLAS_TRANSPOSE trans = get_trans(Ta);
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y, trans);

            #pragma omp parallel for schedule(guided)
            for (size_t i = 0; i < a.num_blocks(); ++i) {
                Shape block_shape = a.get_block_shape(i);
                BLAS_INT m = static_cast<BLAS_INT>(block_shape.first);
                BLAS_INT n = static_cast<BLAS_INT>(block_shape.second);

                const float* Ai = static_cast<const float*>(a.get_block_data(i));
                const float* xi = x.data() + a.get_col_offsets()[i];
                float* yi = y.data() + a.get_row_offsets()[i];

                cblas_sgemv(major, trans, m, n, alpha, Ai, m, xi, inx, beta, yi, iny);
            }
        }

        /*! @brief Simple interface to DSYMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        for specified stride
            @param[in] alpha double value by which A*x is scaled
            @param[in] a pointer to the A matrix in column-major ordering
            @param[in] x pointer to the Vector_ x
            @param[in] incx stride of Vector_ x
            @param[in] beta double value by which y is scaled
            @param[in,out] y pointer to the y Vector_ values
            @param[in] incy stride of Vector_ y
        */
        void SymMatrixVectorProduct(const double alpha, const Matrix_<double> &a,
                                    const Vector_<double> &x, const size_t incx, const double beta, Vector_<double> &y, const size_t incy)
        {
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y);
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;

            BLAS_INT lda = get_leading(nrow, ncol);

            cblas_dsymv(major, tri, ncol, alpha, a.data(), lda, x.data(), inx, beta, y.data(), iny);
        };

        /*! @brief Simple interface to DSYMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        for specified stride
            @param[in] a pointer to the A matrix in column-major ordering
            @param[in] x pointer to the Vector_ x
            @param[in] incx stride of Vector_ x
            @param[in,out] y pointer to the y Vector_ values
            @param[in] alpha (optional, default 1.0) double value by which A*x is scaled
            @param[in] beta (optional, default 0.0) double value by which y is scaled
            @param[in] incy (optional, default 1) stride of Vector_ y
        */
        void SymMatrixVectorProduct(const Matrix_<double> &a, const Vector_<double> &x, Vector_<double> &y,
                                    const double alpha, const double beta, const size_t incx, const size_t incy)
        {
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y);
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;

            BLAS_INT lda = get_leading(nrow, ncol);

            cblas_dsymv(major, tri, ncol, alpha, a.data(), lda, x.data(), inx, beta, y.data(), iny);
        };

        /*! @brief Simple interface to DSYMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        for specified stride
            @param[in] alpha float value by which A*x is scaled
            @param[in] a pointer to the A matrix in column-major ordering
            @param[in] x pointer to the Vector_ x
            @param[in] incx stride of Vector_ x
            @param[in] beta float value by which y is scaled
            @param[in,out] y pointer to the y Vector_ values
            @param[in] incy stride of Vector_ y
        */
        void SymMatrixVectorProduct(const float alpha, const Matrix_<float> &a,
                                    const Vector_<float> &x, const size_t incx, const float beta, Vector_<float> &y, const size_t incy)
        {
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y);
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;

            BLAS_INT lda = get_leading(nrow, ncol);

            cblas_ssymv(major, tri, ncol, alpha, a.data(), lda, x.data(), inx, beta, y.data(), iny);
        };

        /*! @brief Simple interface to SSYMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        for specified stride
            @param[in] a pointer to the A matrix in column-major ordering
            @param[in] x pointer to the Vector_ x
            @param[in] incx stride of Vector_ x
            @param[in,out] y pointer to the y Vector_ values
            @param[in] alpha (optional, default 1.0) float value by which A*x is scaled
            @param[in] beta (optional, default 0.0) float value by which y is scaled
            @param[in] incy (optional, default 1) stride of Vector_ y
        */
        void SymMatrixVectorProduct(const Matrix_<float> &a, const Vector_<float> &x, Vector_<float> &y,
                                    const float alpha, const float beta, const size_t incx, const size_t incy)
        {
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y);
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;

            BLAS_INT lda = get_leading(nrow, ncol);

            cblas_ssymv(major, tri, ncol, alpha, a.data(), lda, x.data(), inx, beta, y.data(), iny);
        };

        /*! @brief Simple interface to DSYMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        for specified stride
            @param[in] alpha double value by which A*x is scaled
            @param[in] a pointer to the A matrix in column-major ordering
            @param[in] x pointer to the Vector_ x
            @param[in] incx stride of Vector_ x
            @param[in] beta double value by which y is scaled
            @param[in,out] y pointer to the y Vector_ values
            @param[in] incy stride of Vector_ y
        */
        void SymMatrixVectorProduct(const double alpha, const LowTriMatrix_<double> &a,
                                    const Vector_<double> &x, const size_t incx, const double beta, Vector_<double> &y, const size_t incy)
        {
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y);
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;

            cblas_dspmv(major, tri, ncol, alpha, a.data(), x.data(), inx, beta, y.data(), iny);
        };

        /*! @brief Simple interface to DSYMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        for specified stride
            @param[in] a pointer to the A matrix in column-major ordering
            @param[in] x pointer to the Vector_ x
            @param[in] incx stride of Vector_ x
            @param[in,out] y pointer to the y Vector_ values
            @param[in] alpha (optional, default 1.0) double value by which A*x is scaled
            @param[in] beta (optional, default 0.0) double value by which y is scaled
            @param[in] incy (optional, default 1) stride of Vector_ y
        */
        void SymMatrixVectorProduct(const LowTriMatrix_<double> &a, const Vector_<double> &x, Vector_<double> &y,
                                    const double alpha, const double beta, const size_t incx, const size_t incy)
        {
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y);
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;

            cblas_dspmv(major, tri, ncol, alpha, a.data(), x.data(), inx, beta, y.data(), iny);
        };

        /*! @brief Simple interface to DSYMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        for specified stride
            @param[in] alpha float value by which A*x is scaled
            @param[in] a pointer to the A matrix in column-major ordering
            @param[in] x pointer to the Vector_ x
            @param[in] incx stride of Vector_ x
            @param[in] beta float value by which y is scaled
            @param[in,out] y pointer to the y Vector_ values
            @param[in] incy stride of Vector_ y
        */
        void SymMatrixVectorProduct(const float alpha, const LowTriMatrix_<float> &a,
                                    const Vector_<float> &x, const size_t incx, const float beta, Vector_<float> &y, const size_t incy)
        {
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y);
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;

            cblas_sspmv(major, tri, ncol, alpha, a.data(), x.data(), inx, beta, y.data(), iny);
        };

        /*! @brief Simple interface to SSYMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        for specified stride
            @param[in] a pointer to the A matrix in column-major ordering
            @param[in] x pointer to the Vector_ x
            @param[in] incx stride of Vector_ x
            @param[in,out] y pointer to the y Vector_ values
            @param[in] alpha (optional, default 1.0) float value by which A*x is scaled
            @param[in] beta (optional, default 0.0) float value by which y is scaled
            @param[in] incy (optional, default 1) stride of Vector_ y
        */
        void SymMatrixVectorProduct(const LowTriMatrix_<float> &a, const Vector_<float> &x, Vector_<float> &y,
                                    const float alpha, const float beta, const size_t incx, const size_t incy)
        {
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y);
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;

            cblas_sspmv(major, tri, ncol, alpha, a.data(), x.data(), inx, beta, y.data(), iny);
        };

        /*! @brief Simple interface to DTPMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        for specified stride
            @param[in] T character representing the equation execute "N", "T", or "C"
            @param[in] unit is the given traingular matrix also a unit matrix
            @param[in] a pointer to the lower triangular matrix A in column-major ordering
            @param[in,out] x pointer to the Vector_ x
            @param[in] incx stride of Vector_ x
        */
        void LowTriMatrixVectorProduct(const char *T, const CBLAS_DIAG unit, const LowTriMatrix_<double> &a, Vector_<double> &x, const size_t incx)
        {
            BLAS_INT inx = incx;
            CBLAS_TRANSPOSE trans = get_trans(T);

            int row, col;
            std::tie(row, col) = check_size_mv(a, x);

            cblas_dtpmv(major, tri, trans, unit, col, a.data(), x.data(), inx);
        };

        /*! @brief Simple interface to DTPMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        for specified stride

            @param[in] a pointer to the lower triangular matrix A in column-major ordering
            @param[in,out] x pointer to the Vector_ x
            @param[in] T (optional, default "N") character representing the equation execute "N", "T", or "C"
            @param[in] unit (optional, default CblasNonUnit) is the given traingular matrix also a unit matrix
            @param[in] incx (optional, defaul 1) stride of Vector_ x
        */
        void LowTriMatrixVectorProduct(const LowTriMatrix_<double> &a, Vector_<double> &x, const char *T, const CBLAS_DIAG unit, const size_t incx)
        {
            BLAS_INT inx = incx;
            CBLAS_TRANSPOSE trans = get_trans(T);

            int row, col;
            std::tie(row, col) = check_size_mv(a, x);

            cblas_dtpmv(major, tri, trans, unit, col, a.data(), x.data(), inx);
        };

        /*! @brief Simple interface to STPMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        for specified stride
            @param[in] T character representing the equation execute "N", "T", or "C"
            @param[in] unit is the given traingular matrix also a unit matrix
            @param[in] a pointer to the lower triangular matrix A in column-major ordering
            @param[in,out] x pointer to the Vector_ x
            @param[in] incx stride of Vector_ x
        */
        void LowTriMatrixVectorProduct(const char *T, const CBLAS_DIAG unit, const LowTriMatrix_<float> &a, Vector_<float> &x, const size_t incx)
        {
            BLAS_INT inx = incx;
            CBLAS_TRANSPOSE trans = get_trans(T);

            int row, col;
            std::tie(row, col) = check_size_mv(a, x);

            cblas_stpmv(major, tri, trans, unit, col, a.data(), x.data(), inx);
        };

        /*! @brief Simple interface to STPMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        for specified stride
            @param[in] a pointer to the lower triangular matrix A in column-major ordering
            @param[in,out] x pointer to the Vector_ x
            @param[in] T (optional, default "N") character representing the equation execute "N", "T", or "C"
            @param[in] unit (optional, default CblasNonUnit) is the given traingular matrix also a unit matrix
            @param[in] incx (optional, defaul 1) stride of Vector_ x
        */
        void LowTriMatrixVectorProduct(const LowTriMatrix_<float> &a, Vector_<float> &x, const char *T, CBLAS_DIAG unit, const size_t incx)
        {
            BLAS_INT inx = incx;
            CBLAS_TRANSPOSE trans = get_trans(T);

            int row, col;
            std::tie(row, col) = check_size_mv(a, x);

            cblas_stpmv(major, tri, trans, unit, col, a.data(), x.data(), inx);
        };
    }
}