#include "impl/gpu/additional-level2.hpp"
#include "../../gpu-utils/utils.hpp"
namespace tcgmtensor
{
    namespace gpu
    {
        template<>
        void AddMatrices<double>(const CudaRuntime& cudart, const char* Ta, const char* Tb, const double alpha, const Matrix<double>& a, const Matrix<double>& b,
        const double beta, Matrix<double>& c)
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
            get_cublas_error(cublasDgeam(cudart.handle, transa, transb, m, n, &alpha, a.gpu_data(), lda, &beta, b.gpu_data(), ldb, c.gpu_data(), ldc));
        };

        template<>
        void AddMatrices<float>(const CudaRuntime& cudart, const char* Ta, const char* Tb, const float alpha, const Matrix<float>& a, const Matrix<float>& b,
        const float beta, Matrix<float>& c)
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
            get_cublas_error(cublasSgeam(cudart.handle, transa, transb, m, n, &alpha, a.gpu_data(), lda, &beta, b.gpu_data(), ldb, c.gpu_data(), ldc));
        };

        template<>
        void AddMatrices<double>(const CudaRuntime& cudart, const Matrix<double>& a, const Matrix<double>& b, Matrix<double>& c,
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
            get_cublas_error(cublasDgeam(cudart.handle, transa, transb, m, n, &alpha, a.gpu_data(), lda, &beta, b.gpu_data(), ldb, c.gpu_data(), ldc));
        };

        template<>
        void AddMatrices<float>(const CudaRuntime& cudart, const Matrix<float>& a, const Matrix<float>& b, Matrix<float>& c,
        const float alpha , const float beta, const char* Ta, const char* Tb)
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
            get_cublas_error(cublasSgeam(cudart.handle, transa, transb, m, n, &alpha, a.gpu_data(), lda, &beta, b.gpu_data(), ldb, c.gpu_data(), ldc));
        };
    }
}
