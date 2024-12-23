#include "impl/blas/gpu/level3.hpp"
#include "../../gpu-utils/utils.hpp"

namespace tcgmtensor
{
    namespace gpu{

    void MatrixMatrixProduct(const CudaRuntime& cudart, const char* Ta, const char* Tb, const double alpha, const Matrix_<double>& a, 
                             const Matrix_<double>& b, const double beta, Matrix_<double>& c)
    {
        cublasOperation_t transa = get_trans(Ta);
        cublasOperation_t transb = get_trans(Tb);

        check_device_alloc( cudart, a);
        check_device_alloc( cudart, b);
        check_device_alloc( cudart, c);

        auto [m, n, k] = check_size_mm(a, b, c, transa, transb);

        size_t lda = get_leading(m, k);
        size_t ldb = get_leading(k, n);
        size_t ldc = get_leading(m, n);

        cudart.cublasSetStream_();
        cublasStatus_t istat =  cublasDgemm(cudart.handle, transa, transb, m, n, k, &alpha, a.gpu_data(), lda, b.gpu_data(), ldb, &beta, c.gpu_data(), ldc);
        get_cublas_error(istat);
    };

    void MatrixMatrixProduct(const CudaRuntime& cudart, const Matrix_<double>& a, const Matrix_<double>& b, Matrix_<double>& c,
                             const double alpha, const double beta, const char* Ta, const char* Tb)
    {
        cublasOperation_t transa = get_trans(Ta);
        cublasOperation_t transb = get_trans(Tb);

        check_device_alloc( cudart, a);
        check_device_alloc( cudart, b);
        check_device_alloc( cudart, c);

        auto [m, n, k] = check_size_mm(a, b, c, transa, transb);

        size_t lda = get_leading(m, k);
        size_t ldb = get_leading(k, n);
        size_t ldc = get_leading(m, n);

        cudart.cublasSetStream_();
        cublasStatus_t istat =  cublasDgemm(cudart.handle, transa, transb, m, n, k, &alpha, a.gpu_data(), lda, b.gpu_data(), ldb, &beta, c.gpu_data(), ldc);
        get_cublas_error(istat);
    };
                             
    void MatrixMatrixProduct(const CudaRuntime& cudart, const char* Ta, const char* Tb, const float alpha, const Matrix_<float>& a, const Matrix_<float>& b,
                             const float beta, Matrix_<float>& c)
    {
        cublasOperation_t transa = get_trans(Ta);
        cublasOperation_t transb = get_trans(Tb);

        check_device_alloc( cudart, a);
        check_device_alloc( cudart, b);
        check_device_alloc( cudart, c);

        auto [m, n, k] = check_size_mm(a, b, c, transa, transb);

        size_t lda = get_leading(m, k);
        size_t ldb = get_leading(k, n);
        size_t ldc = get_leading(m, n);

        cudart.cublasSetStream_();
        cublasStatus_t istat =  cublasSgemm(cudart.handle, transa, transb, m, n, k, &alpha, a.gpu_data(), lda, b.gpu_data(), ldb, &beta, c.gpu_data(), ldc);
        get_cublas_error(istat);
    };

    void MatrixMatrixProduct(const CudaRuntime& cudart, const Matrix_<float>& a, const Matrix_<float>& b, Matrix_<float>& c,
                             const float alpha, const float beta, const char* Ta, const char* Tb)
    {
        cublasOperation_t transa = get_trans(Ta);
        cublasOperation_t transb = get_trans(Tb);

        check_device_alloc( cudart, a);
        check_device_alloc( cudart, b);
        check_device_alloc( cudart, c);

        auto [m, n, k] = check_size_mm(a, b, c, transa, transb);

        size_t lda = get_leading(m, k);
        size_t ldb = get_leading(k, n);
        size_t ldc = get_leading(m, n);

        cudart.cublasSetStream_();
        cublasStatus_t istat = cublasSgemm(cudart.handle, transa, transb, m, n, k, &alpha, a.gpu_data(), lda, b.gpu_data(), ldb, &beta, c.gpu_data(), ldc);
        get_cublas_error(istat);
    };

    void SymMatrixMatrixProduct(const CudaRuntime& cudart, const cublasSideMode_t side, const double alpha, const Matrix_<double>& a, const Matrix_<double>& b,
                                const double beta, Matrix_<double>& c)
    {
        auto [m, n, k] = check_size_mm(a, b, c);

        size_t lda = get_leading(m, k);
        size_t ldb = get_leading(k, n);
        size_t ldc = get_leading(m, n);

        check_device_alloc( cudart, a);
        check_device_alloc( cudart, b);
        check_device_alloc( cudart, c);

        cudart.cublasSetStream_();
        cublasStatus_t istat = cublasDsymm(cudart.handle, side, tri_gpu, m, n, &alpha, a.gpu_data(), lda, b.gpu_data(), ldb, &beta, c.gpu_data(), ldc);
        get_cublas_error(istat);
    };

    void SymMatrixMatrixProduct(const CudaRuntime& cudart, const Matrix_<double>& a, const Matrix_<double>& b, Matrix_<double>& c,
                                const double alpha, const double beta, const cublasSideMode_t side)
    {
        auto [m, n, k] = check_size_mm(a, b, c);

        size_t lda = get_leading(m, k);
        size_t ldb = get_leading(k, n);
        size_t ldc = get_leading(m, n);

        check_device_alloc( cudart, a);
        check_device_alloc( cudart, b);
        check_device_alloc( cudart, c);

        cudart.cublasSetStream_();
        cublasStatus_t istat = cublasDsymm(cudart.handle, side, tri_gpu, m, n, &alpha, a.gpu_data(), lda, b.gpu_data(), ldb, &beta, c.gpu_data(), ldc);
        get_cublas_error(istat);
    };

    void SymMatrixMatrixProduct(const CudaRuntime& cudart, const cublasSideMode_t side, const float alpha, const Matrix_<float>& a, const Matrix_<float>& b,
                                const float beta, Matrix_<float>& c)
    {
        auto [m, n, k] = check_size_mm(a, b, c);

        size_t lda = get_leading(m, k);
        size_t ldb = get_leading(k, n);
        size_t ldc = get_leading(m, n);

        check_device_alloc( cudart, a);
        check_device_alloc( cudart, b);
        check_device_alloc( cudart, c);

        cudart.cublasSetStream_();
        cublasStatus_t istat = cublasSsymm(cudart.handle, side, tri_gpu, m, n, &alpha, a.gpu_data(), lda, b.gpu_data(), ldb, &beta, c.gpu_data(), ldc);
        get_cublas_error(istat);
    };

    void SymMatrixMatrixProduct(const CudaRuntime& cudart, const Matrix_<float>& a, const Matrix_<float>& b, Matrix_<float>& c,
                                const float alpha, const float beta, const cublasSideMode_t side)
    {
        auto [m, n, k] = check_size_mm(a, b, c);

        size_t lda = get_leading(m, k);
        size_t ldb = get_leading(k, n);
        size_t ldc = get_leading(m, n);

        check_device_alloc( cudart, a);
        check_device_alloc( cudart, b);
        check_device_alloc( cudart, c);

        cudart.cublasSetStream_(); 
        cublasStatus_t istat = cublasSsymm(cudart.handle, side, tri_gpu, m, n, &alpha, a.gpu_data(), lda, b.gpu_data(), ldb, &beta, c.gpu_data(), ldc);
        get_cublas_error(istat);
    }; 
    }
} // namespace tcgmtensor
