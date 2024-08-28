#include "impl/gpu/additional-level2.hpp"
#include "../../gpu-utils/utils.hpp"

namespace tcgmtensor{
    namespace gpu
    {
        template<>
        void PackedTri2TriMatrix<double>(const CudaRuntime& cudart, const LowTriMatrix<double>& PackedTriMatrix, Matrix<double>& TriMatrix)
        {
            check_device_alloc(cudart, PackedTriMatrix);
            check_device_alloc(cudart, TriMatrix);

            cudart.cublasSetStream_();
            get_cublas_error(cublasDtpttr(cudart.handle, tri_gpu, PackedTriMatrix.shape().first, PackedTriMatrix.gpu_data(), TriMatrix.gpu_data(), TriMatrix.shape().first));
        };

        template<>
        void PackedTri2TriMatrix<float>(const CudaRuntime& cudart, const LowTriMatrix<float>& PackedTriMatrix, Matrix<float>& TriMatrix)
        {
            check_device_alloc(cudart, PackedTriMatrix);
            check_device_alloc(cudart, TriMatrix);

            cudart.cublasSetStream_();
            get_cublas_error(cublasStpttr(cudart.handle, tri_gpu, PackedTriMatrix.shape().first, PackedTriMatrix.gpu_data(), TriMatrix.gpu_data(), TriMatrix.shape().first));
        };
        
        
        template<>
        void TriMatrix2PackedTri<double>(const CudaRuntime& cudart, const Matrix<double>& TriMatrix, LowTriMatrix<double>& PackedTriMatrix)
        {
            check_device_alloc(cudart, PackedTriMatrix);
            check_device_alloc(cudart, TriMatrix);

            cudart.cublasSetStream_();
            get_cublas_error(cublasDtrttp(cudart.handle, tri_gpu, PackedTriMatrix.shape().first, TriMatrix.gpu_data(), TriMatrix.shape().first, PackedTriMatrix.gpu_data()));
        };

         template<>
        void TriMatrix2PackedTri<float>(const CudaRuntime& cudart, const Matrix<float>& TriMatrix, LowTriMatrix<float>& PackedTriMatrix)
        {
            check_device_alloc(cudart, PackedTriMatrix);
            check_device_alloc(cudart, TriMatrix);

            cudart.cublasSetStream_();
            get_cublas_error(cublasStrttp(cudart.handle, tri_gpu, PackedTriMatrix.shape().first, TriMatrix.gpu_data(), TriMatrix.shape().first, PackedTriMatrix.gpu_data()));
        };

        template void PackedTri2TriMatrix<double>(const CudaRuntime&, const LowTriMatrix<double>& PackedTriMatrix, Matrix<double>& TriMatrix);
        template void PackedTri2TriMatrix<float>(const CudaRuntime&, const LowTriMatrix<float>& PackedTriMatrix, Matrix<float>& TriMatrix);
        template void TriMatrix2PackedTri<double>(const CudaRuntime& cudart, const Matrix<double>& TriMatrix, LowTriMatrix<double>& PackedTriMatrix);
        template void TriMatrix2PackedTri<float>(const CudaRuntime& cudart, const Matrix<float>& TriMatrix, LowTriMatrix<float>& PackedTriMatrix);

    } // namespace gpu
}   