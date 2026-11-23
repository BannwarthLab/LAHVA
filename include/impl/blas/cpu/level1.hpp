/// @file level1.hpp
/// @brief C++-style BLAS Level-1 operations (vector-vector operations).
//
// Level-1 BLAS-like operations (vector * vector) - C++ template declarations
// These functions are C++-style wrappers operating on Tensor_<> objects
// They provide convenient overloads for different scalar types and
// dispatch to optimized CPU BLAS kernels. Each overload is provided for double and float
// precision; both precisions are documented explicitly.

#pragma once
#include "const.h"
#include "linalg.hpp"



namespace lahva
{
    namespace cpu
    {
        /// @brief Computes the inner product of two tensors treated as vectors, wrapper to BLAS function <T>dot.
        ///
        /// Performs result = X^T * Y where X and Y are tensors with the same size.
        ///
        /// @tparam T Numerical element type (double, float, complex_double, complex_float).
        /// @param X First input tensor (Tensor_<T>).
        /// @param Y Second input tensor (Tensor_<T>).
        /// @return The inner product of tensors X and Y.
        template <typename T>
        T InnerVectorProduct(const Tensor_<T> &X, const Tensor_<T> &Y);

        /// @brief Computes the inner product of two tensors treated as vectors with strides, wrapper to BLAS function <T>dot.
        ///
        /// Performs result = X^T * Y where X and Y are tensors with the same size and specified strides.
        ///
        /// @tparam T Numerical element type (double, float, complex_double, complex_float).
        /// @param X First input tensor (Tensor_<T>).
        /// @param strideX Stride between consecutive elements in tensor X.
        /// @param Y Second input tensor (Tensor_<T>).
        /// @param strideY Stride between consecutive elements in tensor Y.
        /// @return The inner product of tensors X and Y.
        template <typename T>
        T InnerVectorProduct(const Tensor_<T> &X, const size_t strideX, const Tensor_<T> &Y, const size_t strideY);

        /// @brief Computes the addition of a scaled tensor treated as vector to another tensor, wrapper to BLAS function <T>axpy.
        ///
        /// Performs y = y + a * x where x and y are tensors with specified strides.
        ///
        /// @tparam T Numerical element type (double, float, complex_double, complex_float).
        /// @param a Scalar multiplier for tensor x.
        /// @param x Input tensor x (Tensor_<T>).
        /// @param ix Stride between consecutive elements in tensor x.
        /// @param y Output tensor y (destination / input-output, Tensor_<T>).
        /// @param iy Stride between consecutive elements in tensor y.
        template <typename T>
        void AddVectors(const T a, const Tensor_<T> &x, const size_t ix, Tensor_<T> &y, const size_t iy);

        /// @brief Computes the addition of a scaled tensor treated as vector to another tensor, convenience overload with scalar type conversion.
        ///
        /// Performs y = y + a * x where scalar type is converted to T. Useful for mixed-precision convenience calls.
        ///
        /// @tparam Scalar Scalar type for `a`.
        /// @tparam T Numerical element type (double, float, complex_double, complex_float).
        /// @param a Scalar multiplier for tensor x (will be cast to T).
        /// @param x Input tensor x (Tensor_<T>).
        /// @param ix Stride between consecutive elements in tensor x.
        /// @param y Output tensor y (destination / input-output, Tensor_<T>).
        /// @param iy Stride between consecutive elements in tensor y.
        template <typename Scalar, typename T>
        void AddVectors(const Scalar a, const Tensor_<T> &x, const size_t ix, Tensor_<T> &y, const size_t iy)
        {
            AddVectors((T)a, x, ix, y, iy);
        };

        /// @brief Computes the addition of a scaled tensor treated as vector to another tensor, mixed precision version.
        ///
        /// Performs y = y + a * x where tensors x and y may have different element types. When U == T,
        /// dispatches to the appropriate single-precision or double-precision BLAS function.
        ///
        /// @tparam T Numerical element type for tensor x.
        /// @tparam U Numerical element type for tensor y.
        /// @param a Scalar multiplier for tensor x.
        /// @param x Input tensor x (Tensor_<T>).
        /// @param y Output tensor y (destination / input-output, Tensor_<U>).
        template <typename T, typename U>
        void AddVectors(const double a, const Tensor_<T> &x, Tensor_<U> &y);

        /// @brief Copies elements from one tensor to another, mixed precision version.
        ///
        /// Performs Y = X (with type conversion) where tensors x and y may have different element types.
        /// When U == T, dispatches to the appropriate single-precision or double-precision BLAS function.
        ///
        /// @tparam T Numerical element type for source tensor X.
        /// @tparam U Numerical element type for destination tensor Y.
        /// @param X Source tensor (Tensor_<T>).
        /// @param Y Destination tensor (Tensor_<U>).
        template <typename T, typename U>
        void CopyVectors(const Tensor_<T> &X, Tensor_<U> &Y);

        /// @brief Copies elements from one tensor to another with strides, wrapper to BLAS function <T>copy.
        ///
        /// Performs Y = X where tensors X and Y have the same size and specified strides.
        ///
        /// @tparam T Numerical element type (double, float, complex_double, complex_float).
        /// @param X Source tensor (Tensor_<T>).
        /// @param strideX Stride between consecutive elements in tensor X.
        /// @param Y Destination tensor (Tensor_<T>).
        /// @param strideY Stride between consecutive elements in tensor Y.
        template <typename T>
        void CopyVectors(const Tensor_<T> &X, const size_t strideX, Tensor_<T> &Y, const size_t strideY);

        /// @brief Swaps elements between two tensors treated as vectors, wrapper to BLAS function <T>swap (unit stride version).
        ///
        /// Exchanges elements between tensors X and Y with the same size.
        ///
        /// @tparam T Numerical element type (double, float, complex_double, complex_float).
        /// @param X First tensor (Tensor_<T>).
        /// @param Y Second tensor (Tensor_<T>).
        template <typename T>
        void SwapVectors(Tensor_<T> &X, Tensor_<T> &Y);

        /// @brief Swaps elements between two tensors treated as vectors with strides, wrapper to BLAS function <T>swap.
        ///
        /// Exchanges elements between tensors X and Y with the same size and specified strides.
        ///
        /// @tparam T Numerical element type (double, float, complex_double, complex_float).
        /// @param X First tensor (Tensor_<T>).
        /// @param strideX Stride between consecutive elements in tensor X.
        /// @param Y Second tensor (Tensor_<T>).
        /// @param strideY Stride between consecutive elements in tensor Y.
        template <typename T>
        void SwapVectors(Tensor_<T> &X, const size_t strideX, Tensor_<T> &Y, const size_t strideY);

        /// @brief Scales a tensor treated as vector by a scalar multiplier, wrapper to BLAS function <T>scal.
        ///
        /// Performs x = a * x where x is a tensor.
        ///
        /// @tparam T Numerical element type (double, float, complex_double, complex_float).
        /// @param a Scalar multiplier.
        /// @param x Input tensor x (Tensor_<T>).
        template <typename T>
        void ScaleVector(const T a, Tensor_<T> &x);

        /// @brief Scales a tensor treated as vector by a scalar multiplier, convenience overload with scalar type conversion.
        ///
        /// Performs x = a * x where scalar type is converted to T. Useful for mixed-precision convenience calls.
        ///
        /// @tparam Scalar Scalar type for `a`.
        /// @tparam T Numerical element type (double, float, complex_double, complex_float).
        /// @param a Scalar multiplier (will be cast to T).
        /// @param x Input tensor x (Tensor_<T>).
        template <typename Scalar, typename T>
        void ScaleVector(const Scalar a, Tensor_<T> &x)
        {
            ScaleVector((T)a, x);
        };

        /// @brief Scales a tensor treated as vector by a scalar multiplier with stride, wrapper to BLAS function <T>scal.
        ///
        /// Performs x = a * x where x is a tensor with specified stride.
        ///
        /// @tparam T Numerical element type (double, float, complex_double, complex_float).
        /// @param a Scalar multiplier.
        /// @param x Input tensor x (Tensor_<T>).
        /// @param ix Stride between consecutive elements in tensor x.
        template <typename T>
        void ScaleVector(const T a, Tensor_<T> &x, size_t ix);

        /// @brief Scales a tensor treated as vector by a scalar multiplier with stride, convenience overload with scalar type conversion.
        ///
        /// Performs x = a * x where scalar type is converted to T. Useful for mixed-precision convenience calls.
        ///
        /// @tparam Scalar Scalar type for `a`.
        /// @tparam T Numerical element type (double, float, complex_double, complex_float).
        /// @param a Scalar multiplier (will be cast to T).
        /// @param x Input tensor x (Tensor_<T>).
        /// @param ix Stride between consecutive elements in tensor x.
        template <typename Scalar, typename T>
        void ScaleVector(const Scalar a, Tensor_<T> &x, size_t ix)
        {
            ScaleVector((T)a, x, ix);
        };

        /// @brief Finds the index of the maximum element in a tensor treated as vector, wrapper to BLAS function <T>iamax.
        ///
        /// Returns the index of the element with maximum absolute value in tensor x.
        ///
        /// @tparam T Numerical element type (double, float, complex_double, complex_float).
        /// @param x Input tensor (Tensor_<T>).
        /// @return Index (0-based) of the maximum element in tensor x.
        template <typename T>
        int IndexMaxFromVector(const Tensor_<T> &x);

        /// @brief Finds the index of the maximum element in a tensor treated as vector with stride, wrapper to BLAS function <T>iamax.
        ///
        /// Returns the index of the element with maximum absolute value in tensor x with specified stride.
        ///
        /// @tparam T Numerical element type (double, float, complex_double, complex_float).
        /// @param x Input tensor (Tensor_<T>).
        /// @param ix Stride between consecutive elements in tensor x.
        /// @return Index (0-based) of the maximum element in tensor x.
        template <typename T>
        int IndexMaxFromVector(const Tensor_<T> &x, const size_t ix);

        /// @brief Runtime-dispatching overload that allows passing a CPURuntime first.
        ///
        /// The runtime parameter is ignored for CPU implementations but this overload
        /// mirrors the signatures used for GPU/CUDA dispatching.
        template <typename T>
        T InnerVectorProduct(const CPURuntime &rt_, const Tensor_<T> &X, const Tensor_<T> &Y)
        {
            return InnerVectorProduct(X, Y);
        };

        /// @brief Runtime-dispatching overload that allows passing a CPURuntime first.
        ///
        /// The runtime parameter is ignored for CPU implementations but this overload
        /// mirrors the signatures used for GPU/CUDA dispatching.
        template <typename T>
        T InnerVectorProduct(const CPURuntime &rt_, const Tensor_<T> &X, const size_t strideX, const Tensor_<T> &Y, const size_t strideY)
        {
            return InnerVectorProduct(X, strideX, Y, strideY);
        };

        /// @brief Runtime-dispatching overload that allows passing a CPURuntime first.
        ///
        /// The runtime parameter is ignored for CPU implementations but this overload
        /// mirrors the signatures used for GPU/CUDA dispatching.
        template <typename... Args>
        void AddVectors(const CPURuntime &rt_, Args &&...args)
        {
            (AddVectors(args...));
        }

        /// @brief Runtime-dispatching overload that allows passing a CPURuntime first.
        ///
        /// The runtime parameter is ignored for CPU implementations but this overload
        /// mirrors the signatures used for GPU/CUDA dispatching.
        template <typename... Args>
        void CopyVectors(const CPURuntime &rt_, Args &&...args)
        {
            (CopyVectors(args...));
        }

        /// @brief Runtime-dispatching overload that allows passing a CPURuntime first.
        ///
        /// The runtime parameter is ignored for CPU implementations but this overload
        /// mirrors the signatures used for GPU/CUDA dispatching.
        template <typename... Args>
        void SwapVectors(const CPURuntime &rt_, Args &&...args)
        {
            (SwapVectors(args...));
        }

        /// @brief Runtime-dispatching overload that allows passing a CPURuntime first.
        ///
        /// The runtime parameter is ignored for CPU implementations but this overload
        /// mirrors the signatures used for GPU/CUDA dispatching.
        template <typename... Args>
        void ScaleVector(const CPURuntime &rt_, Args &&...args)
        {
            (ScaleVector(args...));
        }

        /// @brief Runtime-dispatching overload that allows passing a CPURuntime first.
        ///
        /// The runtime parameter is ignored for CPU implementations but this overload
        /// mirrors the signatures used for GPU/CUDA dispatching.
        template <typename... Args>
        int IndexMaxFromVector(const CPURuntime &rt_, Args &&...args)
        {
            return (IndexMaxFromVector(args...));
        }
    }
}
