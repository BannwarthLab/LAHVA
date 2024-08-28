#include "impl/gpu/lapack.hpp"
#include "../../gpu-utils/utils.hpp"
#include "impl/gpu/additional-level2.hpp"

namespace tcgmtensor
{
    namespace gpu
    {
        template<>
        void SolveGenSysLinEquations<double>(CudaRuntime& cudart, const char* Ta, Matrix<double>& a, Matrix<double>& b)
        {

            check_device_alloc(cudart, a);
            check_device_alloc(cudart, b);

            cudaDataType data = CUDA_R_64F; 

            if ( a.shape().first != a.shape().second)
            {
                throw std::runtime_error("A should be symmetric");
            };
            int64_t n = a.shape().first;
            int64_t nrhs = b.shape().second;
            cusolverDnParams_t params;
            size_t wspacedevice;
            size_t wspacehost;
            get_cusolv_error(cusolverDnCreateParams(&params));
            get_cusolv_error(cusolverDnXgetrf_bufferSize(cudart.getcuSolverHandle(), params, n, n, data, a.gpu_data(), n, data, &wspacedevice, &wspacehost));
            
            if (b.shape().first != n)
            {
                throw std::runtime_error("B should have as much rows as A. B(n, nrhs)");
            }
            Vector<int64_t> ipiv(n);
            ipiv.copy2device(cudart);
            void * buffer_device;
            get_cuda_error(cudaMalloc(&buffer_device, wspacedevice));
            void * buffer_host = malloc(wspacehost);
            Vector<int> info(1, 0);
            info.copy2device(cudart);

            get_cusolv_error(cusolverDnXgetrf(cudart.getcuSolverHandle(), params, n, n, data, a.gpu_data(), n, ipiv.gpu_data(),
                                               data, buffer_device, wspacedevice, buffer_host, wspacehost, info.gpu_data()));
            
            get_cuda_error(cudaFree(buffer_device));
            free(buffer_host);

            info.copy2host(cudart);
            if (info.sum() != 0)
            {
                throw std::runtime_error("Failure in DGETRF");
            }
            cublasOperation_t transa = get_trans(Ta);
            info.copy2device(cudart);
            get_cusolv_error(cusolverDnXgetrs(cudart.getcuSolverHandle(), params, transa, n, nrhs, data, a.gpu_data(), n, ipiv.gpu_data(),
                                               data, b.gpu_data(), n, info.gpu_data()));
            info.copy2host(cudart);
            if (info.sum() != 0)
            {
                throw std::runtime_error("Failure in DGETRS");
            }
            
        };

        template<>
        void SolveGenSysLinEquations<float>(CudaRuntime& cudart, const char* Ta, Matrix<float>& a, Matrix<float>& b)
        {

            check_device_alloc(cudart, a);
            check_device_alloc(cudart, b);

            cudaDataType data = CUDA_R_32F; 

            if ( a.shape().first != a.shape().second)
            {
                throw std::runtime_error("A should be symmetric");
            };
            int64_t n = a.shape().first;
            int64_t nrhs = b.shape().second;
            cusolverDnParams_t params;
            size_t wspacedevice;
            size_t wspacehost;
            get_cusolv_error(cusolverDnCreateParams(&params));
            get_cusolv_error(cusolverDnXgetrf_bufferSize(cudart.getcuSolverHandle(), params, n, n, data, a.gpu_data(), n, data, &wspacedevice, &wspacehost));
            
            if (b.shape().first != n)
            {
                throw std::runtime_error("B should have as much rows as A. B(n, nrhs)");
            }
            Vector<int64_t> ipiv(n);
            ipiv.copy2device(cudart);
            void * buffer_device;
            get_cuda_error(cudaMalloc(&buffer_device, wspacedevice));
            void * buffer_host = malloc(wspacehost);
            Vector<int> info(1, 0);
            info.copy2device(cudart);

            get_cusolv_error(cusolverDnXgetrf(cudart.getcuSolverHandle(), params, n, n, data, a.gpu_data(), n, ipiv.gpu_data(),
                                               data, buffer_device, wspacedevice, buffer_host, wspacehost, info.gpu_data()));
            
            get_cuda_error(cudaFree(buffer_device));
            free(buffer_host);

            info.copy2host(cudart);
            if (info.sum() != 0)
            {
                throw std::runtime_error("Failure in DGETRF");
            }
            cublasOperation_t transa = get_trans(Ta);
            info.copy2device(cudart);
            get_cusolv_error(cusolverDnXgetrs(cudart.getcuSolverHandle(), params, transa, n, nrhs, data, a.gpu_data(), n, ipiv.gpu_data(),
                                               data, b.gpu_data(), n, info.gpu_data()));
            info.copy2host(cudart);
            if (info.sum() != 0)
            {
                throw std::runtime_error("Failure in DGETRS");
            }
            
        };

        template<>
        void SolveGenSysLinEquations<double>(CudaRuntime& cudart, Matrix<double>& a, Matrix<double>& b, const char* Ta)
        {

            check_device_alloc(cudart, a);
            check_device_alloc(cudart, b);

            cudaDataType data = CUDA_R_64F; 

            if ( a.shape().first != a.shape().second)
            {
                throw std::runtime_error("A should be symmetric");
            };
            int64_t n = a.shape().first;
            int64_t nrhs = b.shape().second;
            cusolverDnParams_t params;
            size_t wspacedevice;
            size_t wspacehost;
            get_cusolv_error(cusolverDnCreateParams(&params));
            get_cusolv_error(cusolverDnXgetrf_bufferSize(cudart.getcuSolverHandle(), params, n, n, data, a.gpu_data(), n, data, &wspacedevice, &wspacehost));
            
            if (b.shape().first != n)
            {
                throw std::runtime_error("B should have as much rows as A. B(n, nrhs)");
            }
            Vector<int64_t> ipiv(n);
            ipiv.copy2device(cudart);
            void * buffer_device;
            get_cuda_error(cudaMalloc(&buffer_device, wspacedevice));
            void * buffer_host = malloc(wspacehost);
            Vector<int> info(1, 0);
            info.copy2device(cudart);

            get_cusolv_error(cusolverDnXgetrf(cudart.getcuSolverHandle(), params, n, n, data, a.gpu_data(), n, ipiv.gpu_data(),
                                               data, buffer_device, wspacedevice, buffer_host, wspacehost, info.gpu_data()));
            
            get_cuda_error(cudaFree(buffer_device));
            free(buffer_host);

            info.copy2host(cudart);
            if (info.sum() != 0)
            {
                throw std::runtime_error("Failure in DGETRF");
            }
            cublasOperation_t transa = get_trans(Ta);
            info.copy2device(cudart);
            get_cusolv_error(cusolverDnXgetrs(cudart.getcuSolverHandle(), params, transa, n, nrhs, data, a.gpu_data(), n, ipiv.gpu_data(),
                                               data, b.gpu_data(), n, info.gpu_data()));
            info.copy2host(cudart);
            if (info.sum() != 0)
            {
                throw std::runtime_error("Failure in DGETRS");
            }
            
        };

        template<>
        void SolveGenSysLinEquations<float>(CudaRuntime& cudart, Matrix<float>& a, Matrix<float>& b, const char* Ta)
        {

            check_device_alloc(cudart, a);
            check_device_alloc(cudart, b);

            cudaDataType data = CUDA_R_32F; 
            if ( a.shape().first != a.shape().second)
            {
                throw std::runtime_error("A should be symmetric");
            };
            int64_t n = a.shape().first;
            int64_t nrhs = b.shape().second;
            cusolverDnParams_t params;
            size_t wspacedevice;
            size_t wspacehost;
            get_cusolv_error(cusolverDnCreateParams(&params));
            get_cusolv_error(cusolverDnXgetrf_bufferSize(cudart.getcuSolverHandle(), params, n, n, data, a.gpu_data(), n, data, &wspacedevice, &wspacehost));
            
            if (b.shape().first != n)
            {
                throw std::runtime_error("B should have as much rows as A. B(n, nrhs)");
            }
            Vector<int64_t> ipiv(n);
            ipiv.copy2device(cudart);
            void * buffer_device;
            get_cuda_error(cudaMalloc(&buffer_device, wspacedevice));
            void * buffer_host = malloc(wspacehost);
            Vector<int> info(1, 0);
            info.copy2device(cudart);

            get_cusolv_error(cusolverDnXgetrf(cudart.getcuSolverHandle(), params, n, n, data, a.gpu_data(), n, ipiv.gpu_data(),
                                               data, buffer_device, wspacedevice, buffer_host, wspacehost, info.gpu_data()));
            
            get_cuda_error(cudaFree(buffer_device));
            free(buffer_host);

            info.copy2host(cudart);
            if (info.sum() != 0)
            {
                throw std::runtime_error("Failure in DGETRF");
            }
            cublasOperation_t transa = get_trans(Ta);
            info.copy2device(cudart);
            get_cusolv_error(cusolverDnXgetrs(cudart.getcuSolverHandle(), params, transa, n, nrhs, data, a.gpu_data(), n, ipiv.gpu_data(),
                                               data, b.gpu_data(), n, info.gpu_data()));
            info.copy2host(cudart);
            if (info.sum() != 0)
            {
                throw std::runtime_error("Failure in DGETRS");
            }
            
        };
        
        template<>
        void SolveSymSysLinEquations<double>(CudaRuntime& cudart, Matrix<double>& a, Matrix<double>& b)
        {

            check_device_alloc(cudart, a);
            check_device_alloc(cudart, b);

            cudaDataType data = CUDA_R_64F; 

            if ( a.shape().first != a.shape().second)
            {
                throw std::runtime_error("A should be symmetric");
            };
            int64_t n = a.shape().first;
            int64_t nrhs = b.shape().second;
            cusolverDnParams_t params;
            size_t wspacedevice;
            size_t wspacehost;
            Vector<int64_t> ipiv(n);
            ipiv.copy2device(cudart);
            get_cusolv_error(cusolverDnXsytrs_bufferSize(cudart.getcuSolverHandle(), tri_gpu, n, nrhs, data, a.gpu_data(), n,
                                                         ipiv.gpu_data(), data, b.gpu_data(), n, &wspacedevice, &wspacehost));
            
            if (b.shape().first != n)
            {
                throw std::runtime_error("B should have as much rows as A. B(n, nrhs)");
            }
            
            void * buffer_device;
            get_cuda_error(cudaMalloc(&buffer_device, wspacedevice));
            void * buffer_host = malloc(wspacehost);
            Vector<int> info(1, 0);

            info.copy2device(cudart);
            get_cusolv_error(cusolverDnXsytrs(cudart.getcuSolverHandle(), tri_gpu, n, nrhs, data, a.gpu_data(), n, ipiv.gpu_data(),
                                               data, b.gpu_data(), n, buffer_device, wspacedevice, buffer_host, wspacehost, info.gpu_data()));
            get_cuda_error(cudaFree(buffer_device));
            free(buffer_host);
            info.copy2host(cudart);
            if (info.sum() != 0)
            {
                throw std::runtime_error("Failure in DGETRS");
            }
            
        };

        template<>
        void SolveSymSysLinEquations<float>(CudaRuntime& cudart, Matrix<float>& a, Matrix<float>& b)
        {
            check_device_alloc(cudart, a);
            check_device_alloc(cudart, b);

            cudaDataType data = CUDA_R_32F; 

            if ( a.shape().first != a.shape().second)
            {
                throw std::runtime_error("A should be symmetric");
            };
            int64_t n = a.shape().first;
            int64_t nrhs = b.shape().second;
            cusolverDnParams_t params;
            size_t wspacedevice;
            size_t wspacehost;
            Vector<int64_t> ipiv(n);
            ipiv.copy2device(cudart);
            get_cusolv_error(cusolverDnXsytrs_bufferSize(cudart.getcuSolverHandle(), tri_gpu, n, nrhs, data, a.gpu_data(), n, 
                                                         ipiv.gpu_data(), data, b.gpu_data(), n, &wspacedevice, &wspacehost));
            
            if (b.shape().first != n)
            {
                throw std::runtime_error("B should have as much rows as A. B(n, nrhs)");
            }
            
            void * buffer_device;
            get_cuda_error(cudaMalloc(&buffer_device, wspacedevice));
            void * buffer_host = malloc(wspacehost);
            Vector<int> info(1, 0);

            info.copy2device(cudart);
            get_cusolv_error(cusolverDnXsytrs(cudart.getcuSolverHandle(), tri_gpu, n, nrhs, data, a.gpu_data(), n, ipiv.gpu_data(),
                                               data, b.gpu_data(), n, buffer_device, wspacedevice, buffer_host, wspacehost, info.gpu_data()));
            get_cuda_error(cudaFree(buffer_device));
            free(buffer_host);
            info.copy2host(cudart);
            if (info.sum() != 0)
            {
                throw std::runtime_error("Failure in DGETRS");
            }
        };

        template<>
        void SolveSymSysLinEquations<double>(CudaRuntime& cudart, LowTriMatrix<double>& ap, Matrix<double>& b)
        {

            check_device_alloc(cudart, ap);
            check_device_alloc(cudart, b);
            Matrix<double> a = Matrix<double>(ap.shape());
            PackedTri2TriMatrix(cudart, ap, a);

            SolveSymSysLinEquations(cudart, a, b);

            TriMatrix2PackedTri(cudart, a, ap);

        };

        template<>
        void SolveSymSysLinEquations<float>(CudaRuntime& cudart, LowTriMatrix<float>& ap, Matrix<float>& b)
        {

            check_device_alloc(cudart, ap);
            check_device_alloc(cudart, b);
            Matrix<float> a = Matrix<float>(ap.shape());
            PackedTri2TriMatrix(cudart, ap, a);

            SolveSymSysLinEquations(cudart, a, b);

            TriMatrix2PackedTri(cudart, a, ap);
        };

        template void SolveGenSysLinEquations<double>(CudaRuntime&, const char* Ta, Matrix<double>& a, Matrix<double>& b);
        template void SolveGenSysLinEquations<float>(CudaRuntime&, const char* Ta, Matrix<float>& a, Matrix<float>& b);
        template void SolveGenSysLinEquations<double>(CudaRuntime&, Matrix<double>& a, Matrix<double>& b, const char* Ta);
        template void SolveGenSysLinEquations<float>(CudaRuntime&, Matrix<float>& a, Matrix<float>& b, const char* Ta);
        template void SolveSymSysLinEquations<double>(CudaRuntime&, Matrix<double>& a, Matrix<double>& b);
        template void SolveSymSysLinEquations<float>(CudaRuntime&, Matrix<float>& a, Matrix<float>& b);
        template void SolveSymSysLinEquations<double>(CudaRuntime&, LowTriMatrix<double>& a, Matrix<double>& b);
        template void SolveSymSysLinEquations<float>(CudaRuntime&, LowTriMatrix<float>& a, Matrix<float>& b);
    } // namespace gpu
    
} // namespace tcgmtensor
