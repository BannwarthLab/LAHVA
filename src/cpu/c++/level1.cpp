#include "linalg.hpp"
#include "../utils/utils.hpp"
#include "impl/blas/cpu/level1.hpp"
#include "impl/blas/cpu/level1.h"

namespace lahva
{
    namespace cpu
    {
        ///Dot product routines////////////////////////////////////////////////////////////////////

        /*! @brief Take inner product of two Vectors of T
            \param[in] nelemXY number of elements in Vectors X and Y
            \param[in] X Vector
            \param[in] Y Vector
        */
        template<typename T>
        T InnerVectorProduct(const Tensor<T> &X, const Tensor<T> &Y)
        {
            check_equal_size(X, Y);
            return InnerVectorProduct(X.size(), X.data(), Y.data());
        };

        /// @brief Take inner product of two Vectors of T with strides
        /// @tparam T Numerical type of the tensors
        /// @param X Tensor X
        /// @param strideX Stride between elements in tensor X
        /// @param Y Tensor Y
        /// @param strideY Stride between elements in tensor Y
        /// @return The inner product of tensors X and Y
        template<typename T>
        T InnerVectorProduct(const Tensor<T> &X, const size_t strideX, const Tensor<T> &Y, const size_t strideY)
        {
            check_equal_size(X, Y);
            return InnerVectorProduct(X.size(), X.data(), strideX, Y.data(), strideY);
        };

        /// Add routines////////////////////////////////////////////////////////////////////

        /// @brief Computes the addition of a scaled tensor treated as vector to another tensor treated as vector with strides
        /// @tparam T Numerical type of the tensors
        /// @param a scalar multiplier for tensor x
        /// @param x Tensor x
        /// @param ix Stride between elements in tensor x
        /// @param y Tensor y
        /// @param iy Stride between elements in tensor y
        template<typename T>
        void AddVectors(const T a, const Tensor<T> &x, const size_t ix, Tensor<T> &y, const size_t iy)
        {
            check_equal_size(x, y);
            AddVectors(x.size(), a, x.data(), ix, y.data(), iy);
        };

        /// @brief Computes the addition of a scaled tensor treated as vector to another tensor treated as vector with strides, Mixed precision version
        /// @tparam T Numerical type of the tensor x
        /// @tparam U Numerical type of the tensor y
        /// @param a scalar multiplier for tensor x
        /// @param x Tensor x
        /// @param ix Stride between elements in tensor x
        /// @param y Tensor y
        /// @param iy Stride between elements in tensor y
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
        /// @brief Copies elements from one tensor to another, wrapper to BLAS function <T>copy
        /// @tparam T Numerical type of the tensors
        /// @param x Source tensor
        /// @param y Destination tensor
        template<typename T>
        void CopyVectors(const Tensor<T> &x, Tensor<T> &y)
        {
            check_equal_size(x, y);
            CopyVectors(x.size(), x.data(), y.data());
        }

        /// @brief Copies elements from one tensor to another, Mixed precision version
        /// @brief if U == T, wrapper to the single-precision or double-precision version
        /// @tparam T Numerical type of the source tensor
        /// @tparam U Numerical type of the destination tensor
        /// @param x Source tensor
        /// @param y Destination tensor
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

        /// @brief Copies elements from one tensor to another with strides, wrapper to BLAS function <T>copy
        /// @tparam T Numerical type of the tensors
        /// @param x Source tensor x
        /// @param strideX Stride between elements in tensor x
        /// @param y Destination tensor y
        /// @param strideY Stride between elements in tensor y
        template<typename T>
        void CopyVectors(const Tensor<T> &x, size_t ix, Tensor<T> &y, size_t iy)
        {
            check_equal_size(x, y);
            CopyVectors(x.size(), x.data(), ix, y.data(), iy);
        }

        // Swap routines////////////////////////////////////////////////////////////////////
       /// @brief Swap elements of two tensors treated as vectors, wrapper to BLAS function <T>swap, unit stride version
       /// @tparam T Numerical type of the tensors
       /// @param x tensor x
       /// @param y tensor y
       template<typename T>
        void SwapVectors(Tensor<T> &x, Tensor<T> &y)
        {
            check_equal_size(x, y);
            SwapVectors(x.size(), x.data(), y.data());
        }


       /// @brief Swap elements of two tensors treated as vectors, wrapper to BLAS function <T>swap, specified stride version
       /// @tparam T Numerical type of the tensors
       /// @param x tensor x
       /// @param ix stride of tensor x
       /// @param y tensor y
       /// @param iy stride of tensor y
        template<typename T>
        void SwapVectors(Tensor<T> &x, size_t ix, Tensor<T> &y, size_t iy)
        {
            check_equal_size(x, y);
            SwapVectors(x.size(), x.data(), ix, y.data(), iy);
        }

        // Scale routines////////////////////////////////////////////////////////////////////
        /// @brief Scales a tensor treated as vector by a scalar multiplier, wrapper to BLAS function <T>scal
        /// @tparam T Numerical type of the tensor
        /// @param a scalar multiplier for tensor x
        /// @param x Tensor x
        template<typename T>
        void ScaleVector(const T a, Tensor<T> &x)
        {
            ScaleVector(x.size(), a, x.data());
        }

        /// @brief Scales a tensor treated as vector by a scalar multiplier, wrapper to BLAS function <T>scal, specified stride version
        /// @tparam T Numerical type of the tensor
        /// @param a scalar multiplier for tensor x
        /// @param x Tensor x
        /// @param ix stride of tensor x
        template<typename T>
        void ScaleVector(const T a, Tensor<T> &x, size_t ix)
        {
            ScaleVector(x.size(), a, x.data(), ix);
        };

        // Extrema in Vector routines////////////////////////////////////////////////////////
        /// @brief Get index of the element with the maximum absolute value in a tensor treated as vector, wrapper to BLAS function <T>iamax
        /// @tparam T Numerical type of the tensor
        /// @param x Tensor x
        /// @return Index of the element with the maximum absolute value
        template<typename T>
        int IndexMaxFromVector(const Tensor<T> &x)
        {
            return IndexMaxFromVector(x.size(), x.data());
        }


        /// @brief Get index of the element with the maximum absolute value in a tensor treated as vector, wrapper to BLAS function <T>iamax, specified stride version
        /// @tparam T Numerical type of the tensor
        /// @param x Tensor x
        /// @param ix stride of tensor x
        /// @return Index of the element with the maximum absolute value
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