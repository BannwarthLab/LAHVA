#include "linalg.hpp"
#include "runtime.hpp"
#include "impl/blas/gpu/level2.hpp"
#include "../../gpu-utils/utils.hpp"

namespace lahva
{
    namespace gpu
    {
        /*! @brief Simple interface to DGEMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        or \f$\vec{y}=alpha*\mathbf{A}^\intercal*\vec{x}+beta*\vec{y}\f$ or
        \f$\vec{y}=alpha*conj(\mathbf{A}^\intercal)*\vec{x}+beta*\vec{y}\f$ for specified stride
            @param[in] T character representing the equation execute "N", "T", or "C"
            @param[in] alpha double value by which A*x is scaled
            @param[in] a pointer to the A matrix in column-cudart.handle ordering
            @param[in] x pointer to the Vector_ x
            @param[in] incx stride of Vector_ x
            @param[in] beta double value by which y is scaled
            @param[in,out] y pointer to the y Vector_ values
            @param[in] incy stride of Vector_ y
        */
        void MatrixVectorProduct(const CudaRuntime &cudart, const char *T, const double alpha, const Matrix_<double> &a,
                                 const Vector_<double> &x, const size_t incx, const double beta, Vector_<double> &y, const size_t incy)
        {
            size_t inx = incx;
            size_t iny = incy;

            check_device_alloc(cudart, a);
            check_device_alloc(cudart, x);
            check_device_alloc(cudart, y);

            cublasOperation_t trans = get_trans(T);
            auto [nrow, ncol] = check_size_mv(a, x, y, trans);
            size_t lda = get_leading(nrow, ncol);
            cudart.cublasSetStream_();
            cublasStatus_t istat = cublasDgemv(cudart.handle, trans, nrow, ncol, &alpha, a.gpu_data(), lda, x.gpu_data(), inx, &beta, y.gpu_data(), iny);
            get_cublas_error(istat);
        };

        /*! @brief Simple interface to DGEMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
       or \f$\vec{y}=alpha*\mathbf{A}^\intercal*\vec{x}+beta*\vec{y}\f$ or
       \f$\vec{y}=alpha*conj(\mathbf{A}^\intercal)*\vec{x}+beta*\vec{y}\f$ for specified stride
           @param[in] a pointer to the A matrix in column-cudart.handle ordering
           @param[in] x pointer to the Vector_ x
           @param[in,out] y pointer to the y Vector_ values
           @param[in] T (optional, default value "N") character representing the equation execute "N", "T", or "C"
           @param[in] alpha (optional, default 1.0) double value by which A*x is scaled
           @param[in] beta (optional, default 0.0) double value by which y is scaled
           @param[in] incx (optional, default 1) stride of Vector_ x
           @param[in] incy (optional, default 1) stride of Vector_ y
       */
        void MatrixVectorProduct(const CudaRuntime &cudart, const Matrix_<double> &a, const Vector_<double> &x, Vector_<double> &y, const char *T,
                                 const double alpha, const double beta, const size_t incx, const size_t incy)
        {
            size_t inx = incx;
            size_t iny = incy;

            check_device_alloc(cudart, a);
            check_device_alloc(cudart, x);
            check_device_alloc(cudart, y);

            cublasOperation_t trans = get_trans(T);
            auto [nrow, ncol] = check_size_mv(a, x, y, trans);
            size_t lda = get_leading(nrow, ncol);
            cudart.cublasSetStream_();
            cublasStatus_t istat = cublasDgemv(cudart.handle, trans, nrow, ncol, &alpha, a.gpu_data(), lda, x.gpu_data(), inx, &beta, y.gpu_data(), iny);
            get_cublas_error(istat);
        };

        /*! @brief Simple interface to DGEMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        or \f$\vec{y}=alpha*\mathbf{A}^\intercal*\vec{x}+beta*\vec{y}\f$ or
        \f$\vec{y}=alpha*conj(\mathbf{A}^\intercal)*\vec{x}+beta*\vec{y}\f$  for specified stride
            @param[in] T character representing the equation execute "N", "T", or "C"
            @param[in] alpha double value by which A*x is scaled
            @param[in] a pointer to the A matrix in column-cudart.handle ordering
            @param[in] x pointer to the Vector_ x
            @param[in] incx stride of Vector_ x
            @param[in] beta double value by which y is scaled
            @param[in,out] y pointer to the y Vector_ values
            @param[in] incy stride of Vector_ y
        */
        void MatrixVectorProduct(const CudaRuntime &cudart, const char *T, const float alpha, const Matrix_<float> &a,
                                 const Vector_<float> &x, const size_t incx, const float beta, Vector_<float> &y, const size_t incy)
        {
            size_t inx = incx;
            size_t iny = incy;

            check_device_alloc(cudart, a);
            check_device_alloc(cudart, x);
            check_device_alloc(cudart, y);

            cublasOperation_t trans = get_trans(T);
            auto [nrow, ncol] = check_size_mv(a, x, y, trans);
            size_t lda = get_leading(nrow, ncol);
            cudart.cublasSetStream_();
            cublasStatus_t istat = cublasSgemv(cudart.handle, trans, nrow, ncol, &alpha, a.gpu_data(), lda, x.gpu_data(), inx, &beta, y.gpu_data(), iny);
            get_cublas_error(istat);
        };

        /*! @brief Simple interface to SGEMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
       or \f$\vec{y}=alpha*\mathbf{A}^\intercal*\vec{x}+beta*\vec{y}\f$ or
       \f$\vec{y}=alpha*conj(\mathbf{A}^\intercal)*\vec{x}+beta*\vec{y}\f$ for specified stride
           @param[in] a pointer to the A matrix in column-cudart.handle ordering
           @param[in] x pointer to the Vector_ x
           @param[in,out] y pointer to the y Vector_ values
           @param[in] T (optional, default value "N") character representing the equation execute "N", "T", or "C"
           @param[in] alpha (optional, default 1.0) double value by which A*x is scaled
           @param[in] beta (optional, default 0.0) double value by which y is scaled
           @param[in] incx (optional, default 1) stride of Vector_ x
           @param[in] incy (optional, default 1) stride of Vector_ y
       */
        void MatrixVectorProduct(const CudaRuntime &cudart, const Matrix_<float> &a, const Vector_<float> &x, Vector_<float> &y, const char *T,
                                 const float alpha, const float beta, const size_t incx, const size_t incy)
        {
            size_t inx = incx;
            size_t iny = incy;

            check_device_alloc(cudart, a);
            check_device_alloc(cudart, x);
            check_device_alloc(cudart, y);

            cublasOperation_t trans = get_trans(T);
            auto [nrow, ncol] = check_size_mv(a, x, y, trans);
            size_t lda = get_leading(nrow, ncol);
            cudart.cublasSetStream_();
            cublasStatus_t istat = cublasSgemv(cudart.handle, trans, nrow, ncol, &alpha, a.gpu_data(), lda, x.gpu_data(), inx, &beta, y.gpu_data(), iny);
            get_cublas_error(istat);
        };

        /*! @brief Simple interface to DGEMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
       or \f$\vec{y}=alpha*\mathbf{A}^\intercal*\vec{x}+beta*\vec{y}\f$ or
       \f$\vec{y}=alpha*conj(\mathbf{A}^\intercal)*\vec{x}+beta*\vec{y}\f$ for specified stride
           @param[in] T character representing the equation execute "N", "T", or "C"
           @param[in] alpha double value by which A*x is scaled
           @param[in] a pointer to the A matrix in column-cudart.handle ordering
           @param[in] x pointer to the Vector_ x
           @param[in] incx stride of Vector_ x
           @param[in] beta double value by which y is scaled
           @param[in,out] y pointer to the y Vector_ values
           @param[in] incy stride of Vector_ y
       */
        void MatrixVectorProduct(const CudaRuntime &cudart, const char *T, const complex_double alpha, const Matrix_<complex_double> &a,
                                 const Vector_<complex_double> &x, const size_t incx, const complex_double beta, Vector_<complex_double> &y, const size_t incy)
        {
            size_t inx = incx;
            size_t iny = incy;

            check_device_alloc(cudart, a);
            check_device_alloc(cudart, x);
            check_device_alloc(cudart, y);

            cublasOperation_t trans = get_trans(T);
            auto [nrow, ncol] = check_size_mv(a, x, y, trans);
            size_t lda = get_leading(nrow, ncol);
            cudart.cublasSetStream_();
            get_cublas_error(cublasZgemv(cudart.handle, trans, nrow, ncol, reinterpret_cast<const cuDoubleComplex*>(&alpha), 
                                         reinterpret_cast<const cuDoubleComplex*>(a.gpu_data()), lda, reinterpret_cast<const cuDoubleComplex*>(x.gpu_data()), 
                                         inx, reinterpret_cast<const cuDoubleComplex*>(&beta), reinterpret_cast<cuDoubleComplex*>(y.gpu_data()), iny));
        };

        /*! @brief Simple interface to DGEMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        or \f$\vec{y}=alpha*\mathbf{A}^\intercal*\vec{x}+beta*\vec{y}\f$ or
        \f$\vec{y}=alpha*conj(\mathbf{A}^\intercal)*\vec{x}+beta*\vec{y}\f$ for specified stride
        @param[in] a pointer to the A matrix in column-cudart.handle ordering
        @param[in] x pointer to the Vector_ x
        @param[in,out] y pointer to the y Vector_ values
        @param[in] T (optional, default value "N") character representing the equation execute "N", "T", or "C"
        @param[in] alpha (optional, default 1.0) double value by which A*x is scaled
        @param[in] beta (optional, default 0.0) double value by which y is scaled
        @param[in] incx (optional, default 1) stride of Vector_ x
        @param[in] incy (optional, default 1) stride of Vector_ y
        */
        void MatrixVectorProduct(const CudaRuntime &cudart, const Matrix_<complex_double> &a, const Vector_<complex_double> &x, Vector_<complex_double> &y, const char *T,
                                 const complex_double alpha, const complex_double beta, const size_t incx, const size_t incy)
        {
            size_t inx = incx;
            size_t iny = incy;

            check_device_alloc(cudart, a);
            check_device_alloc(cudart, x);
            check_device_alloc(cudart, y);

            cublasOperation_t trans = get_trans(T);
            auto [nrow, ncol] = check_size_mv(a, x, y, trans);
            size_t lda = get_leading(nrow, ncol);
            cudart.cublasSetStream_();
            get_cublas_error(cublasZgemv(cudart.handle, trans, nrow, ncol, reinterpret_cast<const cuDoubleComplex*>(&alpha), 
                                         reinterpret_cast<const cuDoubleComplex*>(a.gpu_data()), lda, reinterpret_cast<const cuDoubleComplex*>(x.gpu_data()), 
                                         inx, reinterpret_cast<const cuDoubleComplex*>(&beta), reinterpret_cast<cuDoubleComplex*>(y.gpu_data()), iny));
        };

        /*! @brief Simple interface to DGEMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        or \f$\vec{y}=alpha*\mathbf{A}^\intercal*\vec{x}+beta*\vec{y}\f$ or
        \f$\vec{y}=alpha*conj(\mathbf{A}^\intercal)*\vec{x}+beta*\vec{y}\f$  for specified stride
        @param[in] T character representing the equation execute "N", "T", or "C"
        @param[in] alpha double value by which A*x is scaled
        @param[in] a pointer to the A matrix in column-cudart.handle ordering
        @param[in] x pointer to the Vector_ x
        @param[in] incx stride of Vector_ x
        @param[in] beta double value by which y is scaled
        @param[in,out] y pointer to the y Vector_ values
        @param[in] incy stride of Vector_ y
        */
        void MatrixVectorProduct(const CudaRuntime &cudart, const char *T, const complex_float alpha, const Matrix_<complex_float> &a,
                                 const Vector_<complex_float> &x, const size_t incx, const complex_float beta, Vector_<complex_float> &y, const size_t incy)
        {
            size_t inx = incx;
            size_t iny = incy;

            check_device_alloc(cudart, a);
            check_device_alloc(cudart, x);
            check_device_alloc(cudart, y);

            cublasOperation_t trans = get_trans(T);
            auto [nrow, ncol] = check_size_mv(a, x, y, trans);
            size_t lda = get_leading(nrow, ncol);
            cudart.cublasSetStream_();
            get_cublas_error(cublasCgemv(cudart.handle, trans, nrow, ncol, reinterpret_cast<const cuComplex*>(&alpha), 
                                         reinterpret_cast<const cuComplex*>(a.gpu_data()), lda, reinterpret_cast<const cuComplex*>(x.gpu_data()), 
                                         inx, reinterpret_cast<const cuComplex*>(&beta), reinterpret_cast<cuComplex*>(y.gpu_data()), iny));
        };

        /*! @brief Simple interface to SGEMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        or \f$\vec{y}=alpha*\mathbf{A}^\intercal*\vec{x}+beta*\vec{y}\f$ or
        \f$\vec{y}=alpha*conj(\mathbf{A}^\intercal)*\vec{x}+beta*\vec{y}\f$ for specified stride
        @param[in] a pointer to the A matrix in column-cudart.handle ordering
        @param[in] x pointer to the Vector_ x
        @param[in,out] y pointer to the y Vector_ values
        @param[in] T (optional, default value "N") character representing the equation execute "N", "T", or "C"
        @param[in] alpha (optional, default 1.0) double value by which A*x is scaled
        @param[in] beta (optional, default 0.0) double value by which y is scaled
        @param[in] incx (optional, default 1) stride of Vector_ x
        @param[in] incy (optional, default 1) stride of Vector_ y
        */
        void MatrixVectorProduct(const CudaRuntime &cudart, const Matrix_<complex_float> &a, const Vector_<complex_float> &x, Vector_<complex_float> &y, const char *T,
                                 const complex_float alpha, const complex_float beta, const size_t incx, const size_t incy)
        {
            size_t inx = incx;
            size_t iny = incy;

            check_device_alloc(cudart, a);
            check_device_alloc(cudart, x);
            check_device_alloc(cudart, y);

            cublasOperation_t trans = get_trans(T);
            auto [nrow, ncol] = check_size_mv(a, x, y, trans);
            size_t lda = get_leading(nrow, ncol);
            cudart.cublasSetStream_();
            get_cublas_error(cublasCgemv(cudart.handle, trans, nrow, ncol, reinterpret_cast<const cuComplex*>(&alpha), 
                                         reinterpret_cast<const cuComplex*>(a.gpu_data()), lda, reinterpret_cast<const cuComplex*>(x.gpu_data()), 
                                         inx, reinterpret_cast<const cuComplex*>(&beta), reinterpret_cast<cuComplex*>(y.gpu_data()), iny));
        };

        /*! @brief Simple interface to DSYMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        for specified stride
            @param[in] alpha double value by which A*x is scaled
            @param[in] a pointer to the A matrix in column-cudart.handle ordering
            @param[in] x pointer to the Vector_ x
            @param[in] incx stride of Vector_ x
            @param[in] beta double value by which y is scaled
            @param[in,out] y pointer to the y Vector_ values
            @param[in] incy stride of Vector_ y
        */
        void SymMatrixVectorProduct(const CudaRuntime &cudart, const double alpha, const Matrix_<double> &a,
                                    const Vector_<double> &x, const size_t incx, const double beta, Vector_<double> &y, const size_t incy)
        {
            auto [nrow, ncol] = check_size_mv(a, x, y);
            size_t inx = incx;
            size_t iny = incy;
            check_device_alloc(cudart, a);
            check_device_alloc(cudart, x);
            check_device_alloc(cudart, y);

            size_t lda = get_leading(nrow, ncol);
            cudart.cublasSetStream_();
            cublasStatus_t istat = cublasDsymv(cudart.handle, tri_gpu, ncol, &alpha, a.gpu_data(), lda, x.gpu_data(), inx, &beta, y.gpu_data(), iny);
            get_cublas_error(istat);
        };

        /*! @brief Simple interface to DSYMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        for specified stride
            @param[in] a pointer to the A matrix in column-cudart.handle ordering
            @param[in] x pointer to the Vector_ x
            @param[in] incx stride of Vector_ x
            @param[in,out] y pointer to the y Vector_ values
            @param[in] alpha (optional, default 1.0) double value by which A*x is scaled
            @param[in] beta (optional, default 0.0) double value by which y is scaled
            @param[in] incy (optional, default 1) stride of Vector_ y
        */
        void SymMatrixVectorProduct(const CudaRuntime &cudart, const Matrix_<double> &a, const Vector_<double> &x, Vector_<double> &y,
                                    const double alpha, const double beta, const size_t incx, const size_t incy)
        {
            auto [nrow, ncol] = check_size_mv(a, x, y);
            size_t inx = incx;
            size_t iny = incy;
            check_device_alloc(cudart, a);
            check_device_alloc(cudart, x);
            check_device_alloc(cudart, y);

            size_t lda = get_leading(nrow, ncol);
            cudart.cublasSetStream_();
            cublasStatus_t istat = cublasDsymv(cudart.handle, tri_gpu, ncol, &alpha, a.gpu_data(), lda, x.gpu_data(), inx, &beta, y.gpu_data(), iny);
            get_cublas_error(istat);
        };

        /*! @brief Simple interface to DSYMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        for specified stride
            @param[in] alpha float value by which A*x is scaled
            @param[in] a pointer to the A matrix in column-cudart.handle ordering
            @param[in] x pointer to the Vector_ x
            @param[in] incx stride of Vector_ x
            @param[in] beta float value by which y is scaled
            @param[in,out] y pointer to the y Vector_ values
            @param[in] incy stride of Vector_ y
        */
        void SymMatrixVectorProduct(const CudaRuntime &cudart, const float alpha, const Matrix_<float> &a,
                                    const Vector_<float> &x, const size_t incx, const float beta, Vector_<float> &y, const size_t incy)
        {
            auto [nrow, ncol] = check_size_mv(a, x, y);
            size_t inx = incx;
            size_t iny = incy;
            check_device_alloc(cudart, a);
            check_device_alloc(cudart, x);
            check_device_alloc(cudart, y);

            size_t lda = get_leading(nrow, ncol);
            cudart.cublasSetStream_();
            cublasStatus_t istat = cublasSsymv(cudart.handle, tri_gpu, ncol, &alpha, a.gpu_data(), lda, x.gpu_data(), inx, &beta, y.gpu_data(), iny);
            get_cublas_error(istat);
        };

        /*! @brief Simple interface to SSYMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        for specified stride
            @param[in] a pointer to the A matrix in column-cudart.handle ordering
            @param[in] x pointer to the Vector_ x
            @param[in] incx stride of Vector_ x
            @param[in,out] y pointer to the y Vector_ values
            @param[in] alpha (optional, default 1.0) float value by which A*x is scaled
            @param[in] beta (optional, default 0.0) float value by which y is scaled
            @param[in] incy (optional, default 1) stride of Vector_ y
        */
        void SymMatrixVectorProduct(const CudaRuntime &cudart, const Matrix_<float> &a, const Vector_<float> &x, Vector_<float> &y,
                                    const float alpha, const float beta, const size_t incx, const size_t incy)
        {
            auto [nrow, ncol] = check_size_mv(a, x, y);
            size_t inx = incx;
            size_t iny = incy;
            check_device_alloc(cudart, a);
            check_device_alloc(cudart, x);
            check_device_alloc(cudart, y);

            size_t lda = get_leading(nrow, ncol);
            cudart.cublasSetStream_();
            cublasStatus_t istat = cublasSsymv(cudart.handle, tri_gpu, ncol, &alpha, a.gpu_data(), lda, x.gpu_data(), inx, &beta, y.gpu_data(), iny);
            get_cublas_error(istat);
        };

        /*! @brief Simple interface to DSYMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        for specified stride
            @param[in] alpha double value by which A*x is scaled
            @param[in] a pointer to the A matrix in column-cudart.handle ordering
            @param[in] x pointer to the Vector_ x
            @param[in] incx stride of Vector_ x
            @param[in] beta double value by which y is scaled
            @param[in,out] y pointer to the y Vector_ values
            @param[in] incy stride of Vector_ y
        */
        void SymMatrixVectorProduct(const CudaRuntime &cudart, const double alpha, const LowTriMatrix_<double> &a,
                                    const Vector_<double> &x, const size_t incx, const double beta, Vector_<double> &y, const size_t incy)
        {
            auto [nrow, ncol] = check_size_mv(a, x, y);
            size_t inx = incx;
            size_t iny = incy;
            check_device_alloc(cudart, a);
            check_device_alloc(cudart, x);
            check_device_alloc(cudart, y);
            cudart.cublasSetStream_();
            cublasStatus_t istat = cublasDspmv(cudart.handle, tri_gpu, ncol, &alpha, a.gpu_data(), x.gpu_data(), inx, &beta, y.gpu_data(), iny);
            get_cublas_error(istat);
        };

        /*! @brief Simple interface to DSYMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        for specified stride
            @param[in] a pointer to the A matrix in column-cudart.handle ordering
            @param[in] x pointer to the Vector_ x
            @param[in] incx stride of Vector_ x
            @param[in,out] y pointer to the y Vector_ values
            @param[in] alpha (optional, default 1.0) double value by which A*x is scaled
            @param[in] beta (optional, default 0.0) double value by which y is scaled
            @param[in] incy (optional, default 1) stride of Vector_ y
        */
        void SymMatrixVectorProduct(const CudaRuntime &cudart, const LowTriMatrix_<double> &a, const Vector_<double> &x, Vector_<double> &y,
                                    const double alpha, const double beta, const size_t incx, const size_t incy)
        {
            auto [nrow, ncol] = check_size_mv(a, x, y);
            size_t inx = incx;
            size_t iny = incy;
            check_device_alloc(cudart, a);
            check_device_alloc(cudart, x);
            check_device_alloc(cudart, y);

            cudart.cublasSetStream_();
            cublasStatus_t istat = cublasDspmv(cudart.handle, tri_gpu, ncol, &alpha, a.gpu_data(), x.gpu_data(), inx, &beta, y.gpu_data(), iny);
            get_cublas_error(istat);
        };

        /*! @brief Simple interface to DSYMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        for specified stride
            @param[in] alpha float value by which A*x is scaled
            @param[in] a pointer to the A matrix in column-cudart.handle ordering
            @param[in] x pointer to the Vector_ x
            @param[in] incx stride of Vector_ x
            @param[in] beta float value by which y is scaled
            @param[in,out] y pointer to the y Vector_ values
            @param[in] incy stride of Vector_ y
        */
        void SymMatrixVectorProduct(const CudaRuntime &cudart, const float alpha, const LowTriMatrix_<float> &a,
                                    const Vector_<float> &x, const size_t incx, const float beta, Vector_<float> &y, const size_t incy)
        {
            auto [nrow, ncol] = check_size_mv(a, x, y);
            size_t inx = incx;
            size_t iny = incy;
            check_device_alloc(cudart, a);
            check_device_alloc(cudart, x);
            check_device_alloc(cudart, y);
            cudart.cublasSetStream_();
            cublasStatus_t istat = cublasSspmv(cudart.handle, tri_gpu, ncol, &alpha, a.gpu_data(), x.gpu_data(), inx, &beta, y.gpu_data(), iny);
            get_cublas_error(istat);
        };

        /*! @brief Simple interface to SSYMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        for specified stride
            @param[in] a pointer to the A matrix in column-cudart.handle ordering
            @param[in] x pointer to the Vector_ x
            @param[in] incx stride of Vector_ x
            @param[in,out] y pointer to the y Vector_ values
            @param[in] alpha (optional, default 1.0) float value by which A*x is scaled
            @param[in] beta (optional, default 0.0) float value by which y is scaled
            @param[in] incy (optional, default 1) stride of Vector_ y
        */
        void SymMatrixVectorProduct(const CudaRuntime &cudart, const LowTriMatrix_<float> &a, const Vector_<float> &x, Vector_<float> &y,
                                    const float alpha, const float beta, const size_t incx, const size_t incy)
        {
            auto [nrow, ncol] = check_size_mv(a, x, y);
            size_t inx = incx;
            size_t iny = incy;
            check_device_alloc(cudart, a);
            check_device_alloc(cudart, x);
            check_device_alloc(cudart, y);

            cudart.cublasSetStream_();
            cublasStatus_t istat = cublasSspmv(cudart.handle, tri_gpu, ncol, &alpha, a.gpu_data(), x.gpu_data(), inx, &beta, y.gpu_data(), iny);
            get_cublas_error(istat);
        };

        ///*! @brief Simple interface to DTPMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        // for specified stride
        //@param[in] T character representing the equation execute "N", "T", or "C"
        //@param[in] unit is the given traingular matrix also a unit matrix
        //@param[in] a pointer to the lower triangular matrix A in column-cudart.handle ordering
        //@param[in,out] x pointer to the Vector_ x
        //@param[in] incx stride of Vector_ x
        //*/
        // void LowTriMatrixVectorProduct(const char* T, const CBLAS_DIAG unit, const LowTriMatrix<double>& a, Vector_<double>& x, const size_t incx)
        //{
        // size_t inx = incx;
        // CBLAS_TRANSPOSE trans = get_trans(T);

        // auto [row, col] = check_size_mv(a, x);

        // cblas_dtpmv(cudart.handle, tri_gpu, trans, unit, col, a.data(), x.data(), inx);
        //};

        ///*! @brief Simple interface to DTPMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        // for specified stride

        //@param[in] a pointer to the lower triangular matrix A in column-cudart.handle ordering
        //@param[in,out] x pointer to the Vector_ x
        //@param[in] T (optional, default "N") character representing the equation execute "N", "T", or "C"
        //@param[in] unit (optional, default CblasNonUnit) is the given traingular matrix also a unit matrix
        //@param[in] incx (optional, defaul 1) stride of Vector_ x
        //*/
        // void LowTriMatrixVectorProduct(const LowTriMatrix<double>& a, Vector_<double>& x, const char* T, const CBLAS_DIAG unit, const size_t incx)
        //{
        // size_t inx = incx;
        // CBLAS_TRANSPOSE trans = get_trans(T);

        // auto [row, col] = check_size_mv(a, x);

        // cblas_dtpmv(cudart.handle, tri_gpu, trans, unit, col, a.data(), x.data(), inx);
        //};

        ///*! @brief Simple interface to STPMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        // for specified stride
        //@param[in] T character representing the equation execute "N", "T", or "C"
        //@param[in] unit is the given traingular matrix also a unit matrix
        //@param[in] a pointer to the lower triangular matrix A in column-cudart.handle ordering
        //@param[in,out] x pointer to the Vector_ x
        //@param[in] incx stride of Vector_ x
        //*/
        // void LowTriMatrixVectorProduct(const char* T, const CBLAS_DIAG unit, const LowTriMatrix<float>& a, Vector_<float>& x, const size_t incx)
        //{
        // size_t inx = incx;
        // CBLAS_TRANSPOSE trans = get_trans(T);

        // auto [row, col] = check_size_mv(a, x);

        // cblas_stpmv(cudart.handle, tri_gpu, trans, unit, col, a.data(), x.data(), inx);
        //};

        ///*! @brief Simple interface to STPMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        // for specified stride
        //@param[in] a pointer to the lower triangular matrix A in column-cudart.handle ordering
        //@param[in,out] x pointer to the Vector_ x
        //@param[in] T (optional, default "N") character representing the equation execute "N", "T", or "C"
        //@param[in] unit (optional, default CblasNonUnit) is the given traingular matrix also a unit matrix
        //@param[in] incx (optional, defaul 1) stride of Vector_ x
        //*/
        // void LowTriMatrixVectorProduct(const LowTriMatrix<float>& a, Vector_<float>& x, const char* T, CBLAS_DIAG unit, const size_t incx)
        //{
        // size_t inx = incx;
        // CBLAS_TRANSPOSE trans = get_trans(T);

        // auto [row, col] = check_size_mv(a, x);

        // cblas_stpmv(cudart.handle, tri_gpu, trans, unit, col, a.data(), x.data(), inx);
        //};
    }
}