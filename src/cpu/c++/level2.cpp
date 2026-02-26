#include "linalg.hpp"
#include "impl/blas/cpu/level2.hpp"
#include "../utils/utils.hpp"
#include "impl/blas/cpu/level2.h"

namespace lahva
{
    namespace cpu
    {
        /// @brief Outer product of two vectors, wrapper to BLAS function <T>ger
        /// @tparam T Numerical type of the vectors
        /// @param x First input vector
        /// @param y Second input vector
        /// @param A Output matrix to store the outer product
        /// @param incx Stride between elements of vector x, default 1
        /// @param incy Stride between elements of vector y, default 1
        /// @param alpha Scalar multiplier for the outer product, default 1.0
        template<typename T>
        void OuterVectorProduct(const Vector<T>& x, const Vector<T>& y, Matrix<T>& A, size_t incx, size_t incy, const T alpha) {
            
            if (A.shape().first != x.size() || A.shape().second != y.size()) {
                throw std::invalid_argument("OuterVectorProduct: Output matrix A has incorrect shape.");
            }

            OuterVectorProduct(x.size(), x.data(), incx, y.size(), y.data(), incy, alpha, A.data());
        }   
       
        template <typename T>
        void MatrixVectorProduct(const char *Ta, const T alpha, const Matrix_<T> &a, const Vector_<T> &x, 
            const size_t incx, const T beta, Vector_<T> &y, const size_t incy)
        {
            CBLAS_TRANSPOSE trans = get_trans(Ta);
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y, trans);

            MatrixVectorProduct(Ta, nrow, ncol, alpha, a.data(), x.data(), incx, beta, y.data(), incy);
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

        // Explicit template instantiations
        template void OuterVectorProduct<double>(const Vector<double>& x, const Vector<double>& y, Matrix<double>& A, size_t incx, size_t incy, const double alpha);
        template void OuterVectorProduct<float>(const Vector<float>& x, const Vector<float>& y, Matrix<float>& A, size_t incx, size_t incy, const float alpha);
        //
        template void MatrixVectorProduct<double>(const char *Ta, const double alpha, const Matrix_<double> &a, const Vector_<double> &x, const size_t incx, const double beta, Vector_<double> &y, const size_t incy);
        template void MatrixVectorProduct<float>(const char *Ta, const float alpha, const Matrix_<float> &a, const Vector_<float> &x, const size_t incx, const float beta, Vector_<float> &y, const size_t incy);
        template void MatrixVectorProduct<complex_double>(const char *Ta, const complex_double alpha, const Matrix_<complex_double> &a, const Vector_<complex_double> &x, const size_t incx, const complex_double beta, Vector_<complex_double> &y, const size_t incy);
        template void MatrixVectorProduct<complex_float>(const char *Ta, const complex_float alpha, const Matrix_<complex_float> &a, const Vector_<complex_float> &x, const size_t incx, const complex_float beta, Vector_<complex_float> &y, const size_t incy);
        //
        


    }
}