/// @file level1.cpp
/// @brief GPU-accelerated Level-1 BLAS operations implementation.
///
/// Implements GPU versions of Level-1 BLAS operations (vector-vector operations)
/// using NVIDIA's cuBLAS library for GPU acceleration. Supports operations like
/// inner products, vector addition, and scalar multiplication on GPU tensors.

#include "../../cpu/utils/utils.hpp"
#include "impl/gpu/utils.hpp"
#include "impl/blas/gpu/level1.hpp"
#include "linalg.hpp"
#include "runtime.hpp"

namespace lahva{
    namespace gpu{
    /// @brief Computes the inner product of two double-precision GPU tensors.
    ///
    /// Performs result = X^T * Y where X and Y are GPU tensors with unit stride.
    /// Uses cuBLAS Ddot function for efficient GPU computation.
    ///
    /// @param cudart CUDA runtime instance.
    /// @param X First double-precision GPU tensor.
    /// @param Y Second double-precision GPU tensor.
    /// @return The inner product of tensors X and Y.
    double InnerVectorProduct(const CudaRuntime& cudart, const GPUTensor_<double>& X, const GPUTensor_<double>& Y)
    {   
        check_equal_size(X,Y);
        check_device_alloc( cudart, X);
        check_device_alloc( cudart, Y);
        
        double result;
        cudart.cublasSetStream_();
        cublasStatus_t istat = cublasDdot(cudart.handle, X.size(), X.gpu_data(), 1, Y.gpu_data(), 1, &result);
        get_cublas_error(istat);
        return result;}

    /// @brief Computes the inner product of two single-precision GPU tensors.
    ///
    /// Performs result = X^T * Y where X and Y are GPU tensors with unit stride.
    /// Uses cuBLAS Sdot function for efficient GPU computation.
    ///
    /// @param cudart CUDA runtime instance.
    /// @param X First single-precision GPU tensor.
    /// @param Y Second single-precision GPU tensor.
    /// @return The inner product of tensors X and Y.
    float InnerVectorProduct(const CudaRuntime& cudart, const GPUTensor_<float>& X, const GPUTensor_<float>& Y)
    { 
        check_equal_size(X,Y);
        check_device_alloc( cudart, X);
        check_device_alloc( cudart, Y);
        
        float result;
        cudart.cublasSetStream_();
        cublasStatus_t istat = cublasSdot(cudart.handle, X.size(), X.gpu_data(), 1, Y.gpu_data(), 1, &result);
        get_cublas_error(istat);
        return result;
    }

    /// @brief Computes the inner product of two double-precision GPU tensors with custom strides.
    ///
    /// Performs result = X^T * Y where X and Y are GPU tensors with specified element strides.
    /// Uses cuBLAS Ddot function for efficient GPU computation with stride support.
    ///
    /// @param cudart CUDA runtime instance.
    /// @param X First double-precision GPU tensor.
    /// @param strideX Memory stride for tensor X.
    /// @param Y Second double-precision GPU tensor.
    /// @param strideY Memory stride for tensor Y.
    /// @return The inner product of tensors X and Y.
    double InnerVectorProduct(const CudaRuntime& cudart, const GPUTensor_<double>& X, const size_t strideX, const GPUTensor_<double>& Y, const size_t strideY)
    {
        check_equal_size(X,Y);
        check_device_alloc( cudart, X);
        check_device_alloc( cudart, Y);
        
        double result;
        cudart.cublasSetStream_();
        cublasStatus_t istat = cublasDdot(cudart.handle, X.size(), X.gpu_data(), strideX, Y.gpu_data(), strideY, &result);
        get_cublas_error(istat);
        return result;
    }

    /// @brief Computes the inner product of two single-precision GPU tensors with custom strides.
    ///
    /// Performs result = X^T * Y where X and Y are GPU tensors with specified element strides.
    /// Uses cuBLAS Sdot function for efficient GPU computation with stride support.
    ///
    /// @param cudart CUDA runtime instance.
    /// @param X First single-precision GPU tensor.
    /// @param strideX Memory stride for tensor X.
    /// @param Y Second single-precision GPU tensor.
    /// @param strideY Memory stride for tensor Y.
    /// @return The inner product of tensors X and Y.
    float InnerVectorProduct(const CudaRuntime& cudart, const GPUTensor_<float>& X, const size_t strideX, const GPUTensor_<float>& Y, const size_t strideY)
    {
        check_equal_size(X,Y);
        check_device_alloc( cudart, X);
        check_device_alloc( cudart, Y);

        float result;
        cudart.cublasSetStream_();
        cublasStatus_t istat = cublasSdot(cudart.handle, X.size(), X.gpu_data(), strideX, Y.gpu_data(), strideY, &result);
        get_cublas_error(istat);
        return result;
    }

    /// @brief Computes scaled vector addition y = alpha * x + y for double-precision tensors (DAXPY).
    ///
    /// Performs in-place scaled vector addition where each element of y is updated by
    /// adding alpha times the corresponding element of x.
    ///
    /// @param cudart CUDA runtime instance.
    /// @param a Scalar factor (alpha) for vector x.
    /// @param x Input double-precision GPU tensor.
    /// @param y Input/output double-precision GPU tensor, replaced with result.
    void AddVectors(const CudaRuntime& cudart, const double a, const GPUTensor_<double>& x, GPUTensor_<double>& y) {
        check_equal_size(x,y);
        check_device_alloc( cudart, x);
        check_device_alloc( cudart, y); 
        cudart.cublasSetStream_();
        cublasStatus_t istat = cublasDaxpy(cudart.handle, x.size(), &a, x.gpu_data(), 1, y.gpu_data(), 1);
        get_cublas_error(istat);
    }

    /// @brief Computes scaled vector addition y = alpha * x + y for single-precision tensors (SAXPY).
    ///
    /// Performs in-place scaled vector addition where each element of y is updated by
    /// adding alpha times the corresponding element of x.
    ///
    /// @param cudart CUDA runtime instance.
    /// @param a Scalar factor (alpha) for vector x.
    /// @param x Input single-precision GPU tensor.
    /// @param y Input/output single-precision GPU tensor, replaced with result.
    void AddVectors(const CudaRuntime& cudart, const float a, const GPUTensor_<float>& x, GPUTensor_<float>& y) {
        check_equal_size(x,y);
        check_device_alloc( cudart, x);
        check_device_alloc( cudart, y); 
        cudart.cublasSetStream_();
        cublasStatus_t istat = cublasSaxpy(cudart.handle, x.size(), &a, x.gpu_data(), 1, y.gpu_data(), 1);
        get_cublas_error(istat); 
    }


    /// @brief Computes scaled vector addition y = a * x + y with custom strides for double-precision tensors (DAXPY).
    ///
    /// Performs in-place scaled vector addition with element strides where y[i*iy] is updated by
    /// adding a times x[i*ix].
    ///
    /// @param cudart CUDA runtime instance.
    /// @param a Scalar factor (alpha) for vector x.
    /// @param x Input double-precision GPU tensor.
    /// @param ix Memory stride for tensor x.
    /// @param y Input/output double-precision GPU tensor, replaced with result.
    /// @param iy Memory stride for tensor y.
    void AddVectors(const CudaRuntime& cudart, const double a, const GPUTensor_<double>& x, size_t ix, GPUTensor_<double>& y, size_t iy) {
        check_equal_size(x,y);
        check_device_alloc( cudart, x);
        check_device_alloc( cudart, y); 
        cudart.cublasSetStream_();
        cublasStatus_t istat = cublasDaxpy(cudart.handle, x.size(), &a, x.gpu_data(), ix, y.gpu_data(), iy);
        get_cublas_error(istat);
    }

    /// @brief Computes scaled vector addition y = a * x + y with custom strides for single-precision tensors (SAXPY).
    ///
    /// Performs in-place scaled vector addition with element strides where y[i*iy] is updated by
    /// adding a times x[i*ix].
    ///
    /// @param cudart CUDA runtime instance.
    /// @param a Scalar factor (alpha) for vector x.
    /// @param x Input single-precision GPU tensor.
    /// @param ix Memory stride for tensor x.
    /// @param y Input/output single-precision GPU tensor, replaced with result.
    /// @param iy Memory stride for tensor y.
    void AddVectors(const CudaRuntime& cudart, const float a, const GPUTensor_<float>& x, size_t ix, GPUTensor_<float>& y, size_t iy) {
        check_equal_size(x,y);
        check_device_alloc( cudart, x);
        check_device_alloc( cudart, y); 
        cudart.cublasSetStream_();
        cublasStatus_t istat = cublasSaxpy(cudart.handle, x.size(), &a, x.gpu_data(), ix, y.gpu_data(), iy);
        get_cublas_error(istat);
    }

    /// @brief Copies double-precision GPU tensor with unit stride (DCOPY).
    ///
    /// Copies all elements from tensor x to tensor y: y = x.
    ///
    /// @param cudart CUDA runtime instance.
    /// @param x Source double-precision GPU tensor.
    /// @param y Destination double-precision GPU tensor.
    void CopyVectors(const CudaRuntime& cudart, const GPUTensor_<double>& x, GPUTensor_<double>& y) {
        check_equal_size(x,y);
        check_device_alloc( cudart, x);
        check_device_alloc( cudart, y); 
        cudart.cublasSetStream_();
        cublasStatus_t istat = cublasDcopy(cudart.handle, x.size(), x.gpu_data(), 1, y.gpu_data(), 1);
        get_cublas_error(istat); 
    }

    /// @brief Copies single-precision GPU tensor with unit stride (SCOPY).
    ///
    /// Copies all elements from tensor x to tensor y: y = x.
    ///
    /// @param cudart CUDA runtime instance.
    /// @param x Source single-precision GPU tensor.
    /// @param y Destination single-precision GPU tensor.
    void CopyVectors(const CudaRuntime& cudart, const GPUTensor_<float>& x, GPUTensor_<float>& y) {
        check_equal_size(x,y);
        check_device_alloc( cudart, x);
        check_device_alloc( cudart, y);
        cudart.cublasSetStream_();
        cublasStatus_t istat = cublasScopy(cudart.handle, x.size(), x.gpu_data(), 1, y.gpu_data(), 1);
        get_cublas_error(istat); 
    }


    /// @brief Copies double-precision GPU tensor with custom strides (DCOPY).
    ///
    /// Copies elements from tensor x to tensor y with specified element strides: y[i*iy] = x[i*ix].
    ///
    /// @param cudart CUDA runtime instance.
    /// @param x Source double-precision GPU tensor.
    /// @param ix Memory stride for tensor x.
    /// @param y Destination double-precision GPU tensor.
    /// @param iy Memory stride for tensor y.
    void CopyVectors(const CudaRuntime& cudart, const GPUTensor_<double>& x, size_t ix, GPUTensor_<double>& y, size_t iy) {
        check_equal_size(x,y);
        check_device_alloc( cudart, x);
        check_device_alloc( cudart, y); 
        cudart.cublasSetStream_();
        cublasStatus_t istat = cublasDcopy(cudart.handle, x.size(), x.gpu_data(), ix, y.gpu_data(), iy);
        get_cublas_error(istat);
    }

    /// @brief Copies single-precision GPU tensor with custom strides (SCOPY).
    ///
    /// Copies elements from tensor x to tensor y with specified element strides: y[i*iy] = x[i*ix].
    ///
    /// @param cudart CUDA runtime instance.
    /// @param x Source single-precision GPU tensor.
    /// @param ix Memory stride for tensor x.
    /// @param y Destination single-precision GPU tensor.
    /// @param iy Memory stride for tensor y.
    void CopyVectors(const CudaRuntime& cudart, const GPUTensor_<float>& x, size_t ix, GPUTensor_<float>& y, size_t iy) {
        check_equal_size(x,y);
        check_device_alloc( cudart, x);
        check_device_alloc( cudart, y); 
        cudart.cublasSetStream_();
        cublasStatus_t istat = cublasScopy(cudart.handle, x.size(), x.gpu_data(), ix, y.gpu_data(), iy);
        get_cublas_error(istat);
    }

    /// @brief Swaps double-precision GPU tensors with unit stride (DSWAP).
    ///
    /// Exchanges elements between tensors x and y in-place.
    ///
    /// @param cudart CUDA runtime instance.
    /// @param x First double-precision GPU tensor.
    /// @param y Second double-precision GPU tensor.
    void SwapVectors(const CudaRuntime& cudart, GPUTensor_<double>& x, GPUTensor_<double>& y) {
        check_equal_size(x,y);
        check_device_alloc( cudart, x);
        check_device_alloc( cudart, y); 
        cudart.cublasSetStream_();
        cublasStatus_t istat = cublasDswap(cudart.handle, x.size(), x.gpu_data(), 1, y.gpu_data(), 1);
        get_cublas_error(istat);
    }

    /// @brief Swaps single-precision GPU tensors with unit stride (SSWAP).
    ///
    /// Exchanges elements between tensors x and y in-place.
    ///
    /// @param cudart CUDA runtime instance.
    /// @param x First single-precision GPU tensor.
    /// @param y Second single-precision GPU tensor.
    void SwapVectors(const CudaRuntime& cudart, GPUTensor_<float>& x, GPUTensor_<float>& y) {
        check_equal_size(x,y);
        check_device_alloc( cudart, x);
        check_device_alloc( cudart, y); 
        
        cudart.cublasSetStream_();
        cublasStatus_t istat = cublasSswap(cudart.handle, x.size(), x.gpu_data(), 1, y.gpu_data(), 1);
        get_cublas_error(istat);
    }

    /// @brief Swaps double-precision GPU tensors with custom strides (DSWAP).
    ///
    /// Exchanges elements between tensors x and y with specified strides in-place: swaps x[i*ix] with y[i*iy].
    ///
    /// @param cudart CUDA runtime instance.
    /// @param x First double-precision GPU tensor.
    /// @param ix Memory stride for tensor x.
    /// @param y Second double-precision GPU tensor.
    /// @param iy Memory stride for tensor y.
    void SwapVectors(const CudaRuntime& cudart, GPUTensor_<double>& x, size_t ix, GPUTensor_<double>& y, size_t iy) {
        check_equal_size(x,y);
        check_device_alloc( cudart, x);
        check_device_alloc( cudart, y); 
        
        cudart.cublasSetStream_();
        cublasStatus_t istat = cublasDswap(cudart.handle, x.size(), x.gpu_data(), ix, y.gpu_data(), iy);
        get_cublas_error(istat);
    }

    /// @brief Swaps single-precision GPU tensors with custom strides (SSWAP).
    ///
    /// Exchanges elements between tensors x and y with specified strides in-place: swaps x[i*ix] with y[i*iy].
    ///
    /// @param cudart CUDA runtime instance.
    /// @param x First single-precision GPU tensor.
    /// @param ix Memory stride for tensor x.
    /// @param y Second single-precision GPU tensor.
    /// @param iy Memory stride for tensor y.
    void SwapVectors(const CudaRuntime& cudart, GPUTensor_<float>& x, size_t ix, GPUTensor_<float>& y, size_t iy) {
        check_equal_size(x,y);
        check_device_alloc( cudart, x);
        check_device_alloc( cudart, y); 
        
        cudart.cublasSetStream_();
        cublasStatus_t istat = cublasSswap(cudart.handle, x.size(), x.gpu_data(), ix, y.gpu_data(), iy);
        get_cublas_error(istat);
    } 

    /// @brief Scales double-precision GPU tensor by scalar factor with unit stride (DSCAL).
    ///
    /// Multiplies all elements of tensor x by scalar a in-place: x = a * x.
    ///
    /// @param cudart CUDA runtime instance.
    /// @param a Scalar factor for multiplication.
    /// @param x Input/output double-precision GPU tensor.
    void ScaleVector(const CudaRuntime& cudart, const double a, GPUTensor_<double>& x) {
        check_device_alloc( cudart, x); 
        
        cudart.cublasSetStream_();
        cublasStatus_t istat = cublasDscal(cudart.handle, x.size(), &a, x.gpu_data(), 1);
        get_cublas_error(istat);
    }
    /// @brief Scales single-precision GPU tensor by scalar factor with unit stride (SSCAL).
    ///
    /// Multiplies all elements of tensor x by scalar a in-place: x = a * x.
    ///
    /// @param cudart CUDA runtime instance.
    /// @param a Scalar factor for multiplication.
    /// @param x Input/output single-precision GPU tensor.
    void ScaleVector(const CudaRuntime& cudart, const float a, GPUTensor_<float>& x){
        check_device_alloc( cudart, x); 
        
        cudart.cublasSetStream_();
        cublasStatus_t istat = cublasSscal(cudart.handle, x.size(), &a, x.gpu_data(), 1);
        get_cublas_error(istat);
    }

    /// @brief Scales double-precision GPU tensor by scalar factor with custom stride (DSCAL).
    ///
    /// Multiplies tensor elements by scalar a in-place with stride: x[i*ix] = a * x[i*ix].
    ///
    /// @param cudart CUDA runtime instance.
    /// @param a Scalar factor for multiplication.
    /// @param x Input/output double-precision GPU tensor.
    /// @param ix Memory stride for tensor x.
    void ScaleVector(const CudaRuntime& cudart, const double a, GPUTensor_<double>& x, size_t ix){
        check_device_alloc( cudart, x);

        cudart.cublasSetStream_();
        cublasStatus_t istat = cublasDscal(cudart.handle, x.size(), &a, x.gpu_data(), ix);
        get_cublas_error(istat);
    }

    /// @brief Scales single-precision GPU tensor by scalar factor with custom stride (SSCAL).
    ///
    /// Multiplies tensor elements by scalar a in-place with stride: x[i*ix] = a * x[i*ix].
    ///
    /// @param cudart CUDA runtime instance.
    /// @param a Scalar factor for multiplication.
    /// @param x Input/output single-precision GPU tensor.
    /// @param ix Memory stride for tensor x.
    void ScaleVector(const CudaRuntime& cudart, const float a, GPUTensor_<float>& x, size_t ix){
        check_device_alloc( cudart, x); 
        
        cudart.cublasSetStream_();
        cublasStatus_t istat = cublasSscal(cudart.handle, x.size(), &a, x.gpu_data(), ix);
        get_cublas_error(istat);
    }
    } // namespace gpu
} // namespace lahva