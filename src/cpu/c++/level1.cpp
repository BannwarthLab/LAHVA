#include "linalg.hpp"
#include "../utils/utils.hpp"
#include "impl/blas/cpu/level1.hpp"
#include "impl/blas/cpu/level1.h"

namespace lahva
{
    namespace cpu
    {
        ///Dot product routines////////////////////////////////////////////////////////////////////

        /// @brief Computes the inner product of two tensors treated as vectors.
        ///
        /// Performs result = X^T * Y where X and Y are tensors with the same size.
        ///
        /// @tparam T Numerical element type (double, float).
        /// @param X First input tensor.
        /// @param Y Second input tensor.
        /// @return The inner product of tensors X and Y.
        template<typename T>
        T InnerVectorProduct(const Tensor<T> &X, const Tensor<T> &Y)
        {
            check_equal_size(X, Y);
            return InnerVectorProduct(X.size(), X.data(), Y.data());
        };

        /// @brief Computes the inner product of two tensors treated as vectors with strides.
        ///
        /// Performs result = X^T * Y where X and Y are tensors with the same size and specified strides.
        ///
        /// @tparam T Numerical element type (double, float).
        /// @param X First input tensor.
        /// @param strideX Stride between consecutive elements in tensor X.
        /// @param Y Second input tensor.
        /// @param strideY Stride between consecutive elements in tensor Y.
        /// @return The inner product of tensors X and Y.
        template<typename T>
        T InnerVectorProduct(const Tensor<T> &X, const size_t strideX, const Tensor<T> &Y, const size_t strideY)
        {
            check_equal_size(X, Y);
            return InnerVectorProduct(X.size(), X.data(), strideX, Y.data(), strideY);
        };

        /// @brief Computes the addition of a scaled tensor to another tensor with specified strides.
        ///
        /// Performs y = y + a * x where x and y are tensors with specified strides.
        ///
        /// @tparam T Numerical element type (double, float).
        /// @param a Scalar multiplier for tensor x.
        /// @param x Input tensor x (not modified).
        /// @param ix Stride between consecutive elements in tensor x.
        /// @param y Input-output tensor y (destination / input-output).
        /// @param iy Stride between consecutive elements in tensor y.
        template<typename T>
        void AddVectors(const T a, const Tensor<T> &x, const size_t ix, Tensor<T> &y, const size_t iy)
        {
            check_equal_size(x, y);
            AddVectors(x.size(), a, x.data(), ix, y.data(), iy);
        };

        /// @brief Computes the addition of a scaled tensor to another tensor (mixed precision version).
        ///
        /// Performs y = y + a * x where tensors x and y may have different element types.
        /// Useful for mixed-precision convenience calls.
        ///
        /// @tparam T Numerical element type of tensor x.
        /// @tparam U Numerical element type of tensor y.
        /// @param a Scalar multiplier for tensor x (cast to T).
        /// @param x Input tensor x (not modified).
        /// @param y Input-output tensor y (destination / input-output).
        template<typename T, typename U>
        void AddVectors(const double a, const Tensor<T> &x, Tensor<U> &y)
        {
            check_equal_size(x, y);
            if constexpr (std::is_same_v<T, U>)
            {
                AddVectors(x.size(), (T)a, x.data(), y.data());
                return;
            }
            else
            {
            #pragma omp parallel for shared(x, y)
            for (int i = 0; i < x.size(); i++)
            {
                y.data()[i] = static_cast<U>(a * x.data()[i] + y.data()[i]);
            }
            }
        }


        // Copy routines////////////////////////////////////////////////////////////////////
        /// @brief Copies elements from one tensor to another.
        ///
        /// Performs element-wise copy: y = x.
        ///
        /// @tparam T Numerical element type of both tensors.
        /// @param x Source tensor (not modified).
        /// @param y Destination tensor (overwritten).
        template<typename T>
        void CopyVectors(const Tensor<T> &x, Tensor<T> &y)
        {
            check_equal_size(x, y);
            CopyVectors(x.size(), x.data(), y.data());
        }

        /// @brief Copies elements from one tensor to another (mixed precision version).
        ///
        /// Performs element-wise type conversion: y[i] = (U)x[i].
        ///
        /// @tparam T Numerical element type of source tensor.
        /// @tparam U Numerical element type of destination tensor.
        /// @param x Source tensor (not modified).
        /// @param y Destination tensor (overwritten).
        template<typename T, typename U>
        void CopyVectors(const Tensor<T> &x, Tensor<U> &y)
        {
            if constexpr (std::is_same_v<T, U>)
            {
                CopyVectors(x.size(), x.data(), y.data());
                return;
            }
            else
            {
                check_equal_size(x, y);
            #pragma omp parallel for shared(x, y)
            for (int i = 0; i < x.size(); i++)
            {
                y.data()[i] = static_cast<U>(x.data()[i]);
            }
        }
        }

        /// @brief Copies elements from one tensor to another with specified strides.
        ///
        /// Performs element-wise copy with strides: y = x.
        ///
        /// @tparam T Numerical element type of both tensors.
        /// @param x Source tensor (not modified).
        /// @param ix Stride between consecutive elements in tensor x.
        /// @param y Destination tensor (overwritten).
        /// @param iy Stride between consecutive elements in tensor y.
        template<typename T>
        void CopyVectors(const Tensor<T> &x, size_t ix, Tensor<T> &y, size_t iy)
        {
            check_equal_size(x, y);
            CopyVectors(x.size(), x.data(), ix, y.data(), iy);
        }

        // Swap routines////////////////////////////////////////////////////////////////////
       /// @brief Swaps elements of two tensors.
       ///
       /// Performs element-wise swap: x <-> y (unit stride version).
       ///
       /// @tparam T Numerical element type of both tensors.
       /// @param x First input-output tensor.
       /// @param y Second input-output tensor.
       template<typename T>
        void SwapVectors(Tensor<T> &x, Tensor<T> &y)
        {
            check_equal_size(x, y);
            SwapVectors(x.size(), x.data(), y.data());
        }


       /// @brief Swaps elements of two tensors with specified strides.
       ///
       /// Performs element-wise swap with strides: x <-> y.
       ///
       /// @tparam T Numerical element type of both tensors.
       /// @param x First input-output tensor.
       /// @param ix Stride between consecutive elements in tensor x.
       /// @param y Second input-output tensor.
       /// @param iy Stride between consecutive elements in tensor y.
        template<typename T>
        void SwapVectors(Tensor<T> &x, size_t ix, Tensor<T> &y, size_t iy)
        {
            check_equal_size(x, y);
            SwapVectors(x.size(), x.data(), ix, y.data(), iy);
        }

        // Scale routines////////////////////////////////////////////////////////////////////
        /// @brief Scales a tensor by a scalar multiplier.
        ///
        /// Performs: x = a * x (unit stride version).
        ///
        /// @tparam T Numerical element type of the tensor.
        /// @param a Scalar multiplier for tensor x.
        /// @param x Input-output tensor to be scaled.
        template<typename T>
        void ScaleVector(const T a, Tensor<T> &x)
        {
            ScaleVector(x.size(), a, x.data());
        }

        /// @brief Scales a tensor by a scalar multiplier with stride.
        ///
        /// Performs: x = a * x with specified stride.
        ///
        /// @tparam T Numerical element type of the tensor.
        /// @param a Scalar multiplier for tensor x.
        /// @param x Input-output tensor to be scaled.
        /// @param ix Stride between consecutive elements in tensor x.
        template<typename T>
        void ScaleVector(const T a, Tensor<T> &x, size_t ix)
        {
            ScaleVector(x.size(), a, x.data(), ix);
        };

        // Extrema in Vector routines////////////////////////////////////////////////////////
        /// @brief Finds the index of the element with the maximum absolute value in a tensor.
        ///
        /// Returns the index of the element with the maximum absolute value.
        ///
        /// @tparam T Numerical element type of the tensor.
        /// @param x Input tensor.
        /// @return Index (0-based) of the element with the maximum absolute value.
        template<typename T>
        int IndexMaxFromVector(const Tensor<T> &x)
        {
            return IndexMaxFromVector(x.size(), x.data());
        }


        /// @brief Finds the index of the element with the maximum absolute value in a tensor with stride.
        ///
        /// Returns the index of the element with the maximum absolute value.
        ///
        /// @tparam T Numerical element type of the tensor.
        /// @param x Input tensor.
        /// @param ix Stride between consecutive elements in tensor x.
        /// @return Index (0-based) of the element with the maximum absolute value.
        template<typename T>
        int IndexMaxFromVector(const Tensor<T> &x, const size_t ix)
        {
            return IndexMaxFromVector(x.size(), x.data(), ix);
        }

        //// Explicit template instantiations
        template double InnerVectorProduct<double>(const Tensor<double> &X, const Tensor<double> &Y);
        template float InnerVectorProduct<float>(const Tensor<float> &X, const Tensor<float> &Y);
        template double InnerVectorProduct<double>(const Tensor<double> &X, const size_t strideX, const Tensor<double> &Y, const size_t strideY);
        template float InnerVectorProduct<float>(const Tensor<float> &X, const size_t strideX, const Tensor<float> &Y, const size_t strideY);
        ////
        template void AddVectors<double>(const double a, const Tensor<double> &x, const size_t ix, Tensor<double> &y, const size_t iy);
        template void AddVectors<float>(const float a, const Tensor<float> &x, const size_t ix, Tensor<float> &y, const size_t iy);
        template void AddVectors<double, double>(const double a, const Tensor<double> &x, Tensor<double> &y);
        template void AddVectors<float, float>(const double a, const Tensor<float> &x, Tensor<float> &y);
        template void AddVectors<double, float>(const double a, const Tensor<double> &x, Tensor<float> &y);
        template void AddVectors<float, double>(const double a, const Tensor<float> &x, Tensor<double> &y);
        ////
        template void CopyVectors<double, double>(const Tensor<double> &x, Tensor<double> &y);
        template void CopyVectors<float, float>(const Tensor<float> &x, Tensor<float> &y);
        template void CopyVectors<double, float>(const Tensor<double> &x, Tensor<float> &y);
        template void CopyVectors<float, double>(const Tensor<float> &x, Tensor<double> &y);
        template void CopyVectors<double>(const Tensor<double> &x, size_t ix, Tensor<double> &y, size_t iy);
        template void CopyVectors<float>(const Tensor<float> &x, size_t ix, Tensor<float> &y, size_t iy);
        ////
        template void SwapVectors<double>(Tensor<double> &x, Tensor<double> &y);
        template void SwapVectors<float>(Tensor<float> &x, Tensor<float> &y);
        template void SwapVectors<double>(Tensor<double> &x, size_t ix, Tensor<double> &y, size_t iy);
        template void SwapVectors<float>(Tensor<float> &x, size_t ix, Tensor<float> &y, size_t iy);
        ////
        template void ScaleVector<double>(const double a, Tensor<double> &x);
        template void ScaleVector<float>(const float a, Tensor<float> &x);
        template void ScaleVector<double>(const double a, Tensor<double> &x, size_t ix);
        template void ScaleVector<float>(const float a, Tensor<float> &x, size_t ix);
        ////
        template int IndexMaxFromVector<double>(const Tensor<double> &x);
        template int IndexMaxFromVector<float>(const Tensor<float> &x);
        template int IndexMaxFromVector<double>(const Tensor<double> &x, const size_t ix);
        template int IndexMaxFromVector<float>(const Tensor<float> &x, const size_t ix);
    }
}