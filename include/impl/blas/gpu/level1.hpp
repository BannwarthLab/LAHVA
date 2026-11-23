/// @file level1.hpp
/// @brief GPU-accelerated BLAS Level-1 operations (vector-vector operations).
///
/// Provides GPU BLAS Level-1 kernel declarations for CUDA acceleration.
/// Level-1 operations include vector dot products, vector scaling, and vector addition.
/// Each function is provided for double and float precision with GPU runtime management.

#pragma once
#include "const.h"
#include "linalg.hpp"
#include "runtime.hpp"

namespace lahva
{
    namespace gpu
    {

    /// @brief Computes the inner product of two double-precision GPU tensors, wrapper to cuBLAS function cublasDdot.
    ///
    /// Performs result = X^T * Y where X and Y are GPU tensors.
    ///
    /// @param cudart CUDA runtime instance
    /// @param X First input GPU tensor (Tensor_<double>).
    /// @param Y Second input GPU tensor (Tensor_<double>).
    /// @return The inner product of tensors X and Y as double.
    double InnerVectorProduct(const CudaRuntime& cudart, const Tensor_<double>& X, const Tensor_<double>& Y);

    /// @brief Computes the inner product of two single-precision GPU tensors, wrapper to cuBLAS function cublasSdot.
    ///
    /// Performs result = X^T * Y where X and Y are GPU tensors.
    ///
    /// @param cudart CUDA runtime instance
    /// @param X First input GPU tensor (Tensor_<float>).
    /// @param Y Second input GPU tensor (Tensor_<float>).
    /// @return The inner product of tensors X and Y as float.
    float InnerVectorProduct(const CudaRuntime& cudart, const Tensor_<float>& X, const Tensor_<float>& Y);

    /// @brief Computes the inner product of two double-precision GPU tensors with strides.
    ///
    /// Performs result = X^T * Y where X and Y are GPU tensors with specified strides.
    ///
    /// @param cudart CUDA runtime instance
    /// @param X First input GPU tensor (Tensor_<double>).
    /// @param strideX Stride between consecutive elements in tensor X.
    /// @param Y Second input GPU tensor (Tensor_<double>).
    /// @param strideY Stride between consecutive elements in tensor Y.
    /// @return The inner product of tensors X and Y as double.
    double InnerVectorProduct(const CudaRuntime& cudart, const Tensor_<double>& X, const size_t strideX, const Tensor_<double>& Y, const size_t strideY);

    /// @brief Computes the inner product of two single-precision GPU tensors with strides.
    ///
    /// Performs result = X^T * Y where X and Y are GPU tensors with specified strides.
    ///
    /// @param cudart CUDA runtime instance
    /// @param X First input GPU tensor (Tensor_<float>).
    /// @param strideX Stride between consecutive elements in tensor X.
    /// @param Y Second input GPU tensor (Tensor_<float>).
    /// @param strideY Stride between consecutive elements in tensor Y.
    /// @return The inner product of tensors X and Y as float.
    float InnerVectorProduct(const CudaRuntime& cudart, const Tensor_<float>& X, const size_t strideX, const Tensor_<float>& Y, const size_t strideY);

    /// @brief Adds scaled double-precision GPU vector to another GPU vector, wrapper to cuBLAS function cublasDaxpy.
    ///
    /// Performs y = y + a * x where a is a scalar multiplier.
    ///
    /// @param cudart CUDA runtime instance
    /// @param a Scalar multiplier for tensor x.
    /// @param x Input GPU tensor x (Tensor_<double>).
    /// @param y Output GPU tensor y (destination / input-output, Tensor_<double>).
    void AddVectors(const CudaRuntime& cudart, double a, const Tensor_<double>& x, Tensor_<double>& y);

    /// @brief Adds scaled single-precision GPU vector to another GPU vector, wrapper to cuBLAS function cublasSaxpy.
    ///
    /// Performs y = y + a * x where a is a scalar multiplier.
    ///
    /// @param cudart CUDA runtime instance
    /// @param a Scalar multiplier for tensor x.
    /// @param x Input GPU tensor x (Tensor_<float>).
    /// @param y Output GPU tensor y (destination / input-output, Tensor_<float>).
    void AddVectors(const CudaRuntime& cudart, float a, const Tensor_<float>& x, Tensor_<float>& y);

    /// @brief Adds scaled double-precision GPU vector with strides to another GPU vector.
    ///
    /// Performs y = y + a * x where x and y have specified strides.
    ///
    /// @param cudart CUDA runtime instance
    /// @param a Scalar multiplier for tensor x.
    /// @param x Input GPU tensor x (Tensor_<double>).
    /// @param ix Stride between consecutive elements in tensor x.
    /// @param y Output GPU tensor y (destination / input-output, Tensor_<double>).
    /// @param iy Stride between consecutive elements in tensor y.
    void AddVectors(const CudaRuntime& cudart, double a, const Tensor_<double>& x, size_t ix, Tensor_<double>& y, size_t iy);

    /// @brief Adds scaled single-precision GPU vector with strides to another GPU vector.
    ///
    /// Performs y = y + a * x where x and y have specified strides.
    ///
    /// @param cudart CUDA runtime instance
    /// @param a Scalar multiplier for tensor x.
    /// @param x Input GPU tensor x (Tensor_<float>).
    /// @param ix Stride between consecutive elements in tensor x.
    /// @param y Output GPU tensor y (destination / input-output, Tensor_<float>).
    /// @param iy Stride between consecutive elements in tensor y.
    void AddVectors(const CudaRuntime& cudart, float a, const Tensor_<float>& x, size_t ix, Tensor_<float>& y, size_t iy);

    /// @brief Adds scaled double-precision GPU vector to single-precision GPU vector (mixed precision).
    ///
    /// Performs y = y + a * x with type conversion from double to float.
    ///
    /// @param cudart CUDA runtime instance
    /// @param a Scalar multiplier for tensor x.
    /// @param x Input GPU tensor x (Tensor_<double>).
    /// @param y Output GPU tensor y (destination / input-output, Tensor_<float>).
    void AddVectors(const CudaRuntime& cudart, double a, const Tensor_<double>& x, Tensor_<float>& y);

    /// @brief Adds scaled single-precision GPU vector to double-precision GPU vector (mixed precision).
    ///
    /// Performs y = y + a * x with type conversion from float to double.
    ///
    /// @param cudart CUDA runtime instance
    /// @param a Scalar multiplier for tensor x.
    /// @param x Input GPU tensor x (Tensor_<float>).
    /// @param y Output GPU tensor y (destination / input-output, Tensor_<double>).
    void AddVectors(const CudaRuntime& cudart, double a, const Tensor_<float>& x, Tensor_<double>& y);

    /// @brief Adds scaled half-precision GPU vector to single-precision GPU vector (mixed precision).
    ///
    /// Performs y = y + a * x with type conversion from half to float.
    ///
    /// @param cudart CUDA runtime instance
    /// @param a Scalar multiplier for tensor x.
    /// @param x Input GPU tensor x (Tensor_<__half>).
    /// @param y Output GPU tensor y (destination / input-output, Tensor_<float>).
    void AddVectors(const CudaRuntime& cudart, float a, const Tensor_<__half>& x, Tensor_<float>& y);

    /// @brief Adds scaled half-precision GPU vector to double-precision GPU vector (mixed precision).
    ///
    /// Performs y = y + a * x with type conversion from half to double.
    ///
    /// @param cudart CUDA runtime instance
    /// @param a Scalar multiplier for tensor x.
    /// @param x Input GPU tensor x (Tensor_<__half>).
    /// @param y Output GPU tensor y (destination / input-output, Tensor_<double>).
    void AddVectors(const CudaRuntime& cudart, double a, const Tensor_<__half>& x, Tensor_<double>& y);

    /// @brief Copies double-precision GPU tensor to another double-precision GPU tensor, wrapper to cuBLAS function cublasDcopy.
    ///
    /// Performs Y = X where both tensors are double-precision.
    ///
    /// @param cudart CUDA runtime instance
    /// @param X Source GPU tensor (Tensor_<double>).
    /// @param Y Destination GPU tensor (Tensor_<double>).
    void CopyVectors(const CudaRuntime& cudart, const Tensor_<double>& X, Tensor_<double>& Y);

    /// @brief Copies single-precision GPU tensor to another single-precision GPU tensor, wrapper to cuBLAS function cublasScopy.
    ///
    /// Performs Y = X where both tensors are single-precision.
    ///
    /// @param cudart CUDA runtime instance
    /// @param X Source GPU tensor (Tensor_<float>).
    /// @param Y Destination GPU tensor (Tensor_<float>).
    void CopyVectors(const CudaRuntime& cudart, const Tensor_<float>& X, Tensor_<float>& Y);

    /// @brief Copies double-precision GPU tensor to single-precision GPU tensor (mixed precision).
    ///
    /// Performs Y = X with type conversion from double to float.
    ///
    /// @param cudart CUDA runtime instance
    /// @param X Source GPU tensor (Tensor_<double>).
    /// @param Y Destination GPU tensor (Tensor_<float>).
    void CopyVectors(const CudaRuntime& cudart, const Tensor_<double>& X, Tensor_<float>& Y);

    /// @brief Copies single-precision GPU tensor to double-precision GPU tensor (mixed precision).
    ///
    /// Performs Y = X with type conversion from float to double.
    ///
    /// @param cudart CUDA runtime instance
    /// @param X Source GPU tensor (Tensor_<float>).
    /// @param Y Destination GPU tensor (Tensor_<double>).
    void CopyVectors(const CudaRuntime& cudart, const Tensor_<float>& X, Tensor_<double>& Y);

    /// @brief Copies double-precision GPU tensor with strides to another double-precision GPU tensor.
    ///
    /// Performs Y = X where both tensors are double-precision with specified strides.
    ///
    /// @param cudart CUDA runtime instance
    /// @param X Source GPU tensor (Tensor_<double>).
    /// @param strideX Stride between consecutive elements in tensor X.
    /// @param Y Destination GPU tensor (Tensor_<double>).
    /// @param strideY Stride between consecutive elements in tensor Y.
    void CopyVectors(const CudaRuntime& cudart, const Tensor_<double>& X, const size_t strideX, Tensor_<double>& Y, const size_t strideY);

    /// @brief Copies single-precision GPU tensor with strides to another single-precision GPU tensor.
    ///
    /// Performs Y = X where both tensors are single-precision with specified strides.
    ///
    /// @param cudart CUDA runtime instance
    /// @param X Source GPU tensor (Tensor_<float>).
    /// @param strideX Stride between consecutive elements in tensor X.
    /// @param Y Destination GPU tensor (Tensor_<float>).
    /// @param strideY Stride between consecutive elements in tensor Y.
    void CopyVectors(const CudaRuntime& cudart, const Tensor_<float>& X, const size_t strideX, Tensor_<float>& Y, const size_t strideY);

    /// @brief Swaps two double-precision GPU vectors.
    ///
    /// Exchanges the contents of tensors X and Y (double-precision).
    ///
    /// @param cudart CUDA runtime instance
    /// @param X First GPU tensor to swap (Tensor_<double>).
    /// @param Y Second GPU tensor to swap (Tensor_<double>).
    void SwapVectors(const CudaRuntime& cudart, Tensor_<double>& X, Tensor_<double>& Y);

    /// @brief Swaps two single-precision GPU vectors.
    ///
    /// Exchanges the contents of tensors X and Y (single-precision).
    ///
    /// @param cudart CUDA runtime instance
    /// @param X First GPU tensor to swap (Tensor_<float>).
    /// @param Y Second GPU tensor to swap (Tensor_<float>).
    void SwapVectors(const CudaRuntime& cudart, Tensor_<float>& X, Tensor_<float>& Y);

    /// @brief Swaps two double-precision GPU vectors with strides.
    ///
    /// Exchanges the contents of tensors X and Y with specified strides.
    ///
    /// @param cudart CUDA runtime instance
    /// @param X First GPU tensor to swap (Tensor_<double>).
    /// @param strideX Stride between consecutive elements in tensor X.
    /// @param Y Second GPU tensor to swap (Tensor_<double>).
    /// @param strideY Stride between consecutive elements in tensor Y.
    void SwapVectors(const CudaRuntime& cudart, Tensor_<double>& X, const size_t strideX, Tensor_<double>& Y, const size_t strideY);

    /// @brief Swaps two single-precision GPU vectors with strides.
    ///
    /// Exchanges the contents of tensors X and Y with specified strides.
    ///
    /// @param cudart CUDA runtime instance
    /// @param X First GPU tensor to swap (Tensor_<float>).
    /// @param strideX Stride between consecutive elements in tensor X.
    /// @param Y Second GPU tensor to swap (Tensor_<float>).
    /// @param strideY Stride between consecutive elements in tensor Y.
    void SwapVectors(const CudaRuntime& cudart, Tensor_<float>& X, const size_t strideX, Tensor_<float>& Y, const size_t strideY);

    /// @brief Scales double-precision GPU vector by a scalar, wrapper to cuBLAS function cublasDscal.
    ///
    /// Performs x = a * x where a is a scalar multiplier.
    ///
    /// @param cudart CUDA runtime instance
    /// @param a Scalar multiplier for tensor x.
    /// @param x GPU tensor to scale in-place (Tensor_<double>).
    void ScaleVector(const CudaRuntime& cudart, double a, Tensor_<double>& x);

    /// @brief Scales single-precision GPU vector by a scalar, wrapper to cuBLAS function cublasSscal.
    ///
    /// Performs x = a * x where a is a scalar multiplier.
    ///
    /// @param cudart CUDA runtime instance
    /// @param a Scalar multiplier for tensor x.
    /// @param x GPU tensor to scale in-place (Tensor_<float>).
    void ScaleVector(const CudaRuntime& cudart, float a, Tensor_<float>& x);

    /// @brief Scales double-precision GPU vector by a scalar with stride.
    ///
    /// Performs x = a * x where a is a scalar multiplier and x has specified stride.
    ///
    /// @param cudart CUDA runtime instance
    /// @param a Scalar multiplier for tensor x.
    /// @param x GPU tensor to scale in-place (Tensor_<double>).
    /// @param ix Stride between consecutive elements in tensor x.
    void ScaleVector(const CudaRuntime& cudart, double a, Tensor_<double>& x, size_t ix);

    /// @brief Scales single-precision GPU vector by a scalar with stride.
    ///
    /// Performs x = a * x where a is a scalar multiplier and x has specified stride.
    ///
    /// @param cudart CUDA runtime instance
    /// @param a Scalar multiplier for tensor x.
    /// @param x GPU tensor to scale in-place (Tensor_<float>).
    /// @param ix Stride between consecutive elements in tensor x.
    void ScaleVector(const CudaRuntime& cudart, float a, Tensor_<float>& x, size_t ix);

    /// @brief Adds GPU tensor with type conversion and precision change (template version).
    ///
    /// Performs out = out + in with conversion from input precision to output precision.
    ///
    /// @tparam inprec Input precision type 
    /// @tparam outprec Output precision type
    /// @param cudart CUDA runtime instance
    /// @param in Input GPU tensor (Tensor_<inprec>).
    /// @param out Output GPU tensor to accumulate into (Tensor_<outprec>).
    template<typename inprec, typename outprec>
    void AddVectors(const CudaRuntime& cudart, const Tensor_<inprec>& in, Tensor_<outprec>& out);

    } // namespace gpu
} // namespace lahva
