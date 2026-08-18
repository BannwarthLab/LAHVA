#include "linalg.hpp"
#include "runtime.hpp"
#include "impl/blas/gpu/additional-level2.hpp"
#include "../gpu-utils/utils.hpp"
namespace lahva
{
    namespace gpu
    {
        template<>
        void AddMatrices<double>(const CudaRuntime& cudart, const char* Ta, const char* Tb, const double alpha, const Matrix_<double>& a, const Matrix_<double>& b,
        const double beta, Matrix_<double>& c)
        {
            cublasOperation_t transa = get_trans(Ta);
            cublasOperation_t transb = get_trans(Tb);

            check_device_alloc( cudart, a);
            check_device_alloc( cudart, b);
            check_device_alloc( cudart, c);

            int m, n;
            std::tie(m, n) = check_same_shape_mm(a, b, c, transa, transb);

            size_t lda = get_leading(m, n, transa);
            size_t ldb = get_leading(m, n, transb);
            size_t ldc = get_leading(m, n);

            cudart.cublasSetStream_();
            get_cublas_error(cublasDgeam(cudart.handle, transa, transb, m, n, &alpha, a.gpu_data(), lda, &beta, b.gpu_data(), ldb, c.gpu_data(), ldc));
        };

        template<>
        void AddMatrices<float>(const CudaRuntime& cudart, const char* Ta, const char* Tb, const float alpha, const Matrix_<float>& a, const Matrix_<float>& b,
        const float beta, Matrix_<float>& c)
        {
            cublasOperation_t transa = get_trans(Ta);
            cublasOperation_t transb = get_trans(Tb);

            check_device_alloc( cudart, a);
            check_device_alloc( cudart, b);
            check_device_alloc( cudart, c);

            int m, n;
            std::tie(m, n) = check_same_shape_mm(a, b, c, transa, transb);

            size_t lda = get_leading(m, n, transa);
            size_t ldb = get_leading(m, n, transb);
            size_t ldc = get_leading(m, n);

            cudart.cublasSetStream_();
            get_cublas_error(cublasSgeam(cudart.handle, transa, transb, m, n, &alpha, a.gpu_data(), lda, &beta, b.gpu_data(), ldb, c.gpu_data(), ldc));
        };

        template<>
        void AddMatrices<double>(const CudaRuntime& cudart, const Matrix_<double>& a, const Matrix_<double>& b, Matrix_<double>& c,
        const double alpha, const double beta, const char* Ta, const char* Tb)
        {
            cublasOperation_t transa = get_trans(Ta);
            cublasOperation_t transb = get_trans(Tb);

            check_device_alloc( cudart, a);
            check_device_alloc( cudart, b);
            check_device_alloc( cudart, c);

            int m, n;
            std::tie(m, n) = check_same_shape_mm(a, b, c, transa, transb);

            size_t lda = get_leading(m, n, transa);
            size_t ldb = get_leading(m, n, transb);
            size_t ldc = get_leading(m, n);

            cudart.cublasSetStream_();
            get_cublas_error(cublasDgeam(cudart.handle, transa, transb, m, n, &alpha, a.gpu_data(), lda, &beta, b.gpu_data(), ldb, c.gpu_data(), ldc));
        };

        template<>
        void AddMatrices<float>(const CudaRuntime& cudart, const Matrix_<float>& a, const Matrix_<float>& b, Matrix_<float>& c,
        const float alpha , const float beta, const char* Ta, const char* Tb)
        {
            cublasOperation_t transa = get_trans(Ta);
            cublasOperation_t transb = get_trans(Tb);

            check_device_alloc( cudart, a);
            check_device_alloc( cudart, b);
            check_device_alloc( cudart, c);

            int m, n;
            std::tie(m, n) = check_same_shape_mm(a, b, c, transa, transb);

            size_t lda = get_leading(m, n, transa);
            size_t ldb = get_leading(m, n, transb);
            size_t ldc = get_leading(m, n);

            cudart.cublasSetStream_();
            get_cublas_error(cublasSgeam(cudart.handle, transa, transb, m, n, &alpha, a.gpu_data(), lda, &beta, b.gpu_data(), ldb, c.gpu_data(), ldc));
        };
    }
}
