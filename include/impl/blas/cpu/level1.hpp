#ifndef LAHVA_C_LEVEL1_HPP
#define LAHVA_C_LEVEL1_HPP
#include "const.h"
#include "linalg.hpp"

// In general, the functions declared here are C++-style BLAS level 1 functions
// operating on Tensor<> objects, meaning vectors and matrices as well as lowtriangular matrices objects
namespace lahva
{
    namespace cpu
    {
        /// @brief Computes the inner product of two tensors treated as vectors, wrapper to BLAS function <T>dot
        /// @tparam T Numerical type of the tensors
        /// @param X Tensor X
        /// @param Y Tensor Y
        /// @return The inner product of tensors X and Y
        template <typename T>
        T InnerVectorProduct(const Tensor<T> &X, const Tensor<T> &Y);

        /// @brief Computes the inner product of two tensors treated as vectors with strides, wrapper to BLAS function <T>dot
        /// @tparam T Numerical type of the tensors
        /// @param X Tensor X
        /// @param strideX Stride between elements in tensor X
        /// @param Y Tensor Y
        /// @param strideY Stride between elements in tensor Y
        /// @return The inner product of tensors X and Y
        template <typename T>
        T InnerVectorProduct(const Tensor<T> &X, const size_t strideX, const Tensor<T> &Y, const size_t strideY);

        /// @brief Wrapper function to compute the inner product of two tensors treated as vectors, with CPURuntime as first argument
        /// @tparam T Numerical type of the tensors
        /// @param rt_ the CPURuntime instance, which is ignored
        /// @param X Tensor X
        /// @param Y Tensor Y
        /// @return The inner product of tensors X and Y
        template <typename T>
        T InnerVectorProduct(const CPURuntime &rt_, const Tensor<T> &X, const Tensor<T> &Y)
        {
            return InnerVectorProduct(X, Y);
        };

        /// @brief Computes the addition of a scaled tensor treated as vector to another tensor treated as vector with strides, wrapper to BLAS function <T>axpy
        /// @tparam T Numerical type of the tensors
        /// @param a scalar multiplier for tensor x
        /// @param x Tensor x
        /// @param ix Stride between elements in tensor x
        /// @param y Tensor y
        /// @param iy Stride between elements in tensor y
        template <typename T>
        void AddVectors(const T a, const Tensor<T> &x, const size_t ix, Tensor<T> &y, const size_t iy);

        /// @brief Computes the addition of a scaled tensor treated as vector to another tensor treated as vector with strides, wrapper to BLAS function <T>axpy
        /// @tparam Scalar type of scalar multiplier
        /// @tparam T Numerical type of the tensors
        /// @param a scalar multiplier for tensor x
        /// @param x Tensor x
        /// @param ix Stride between elements in tensor x
        /// @param y Tensor y
        /// @param iy Stride between elements in tensor y
        template <typename Scalar, typename T>
        void AddVectors(const Scalar a, const Tensor<T> &x, const size_t ix, Tensor<T> &y, const size_t iy)
        {
            AddVectors((T)a, x, ix, y, iy);
        };

        /// @brief Computes the addition of a scaled tensor treated as vector to another tensor treated as vector, Mixed precision version
        /// @brief if U == T, wrapper to the single-precision or double-precision version
        /// @tparam T Numerical type of the tensor x
        /// @tparam U Numerical type of the tensor y
        /// @param a scalar multiplier for tensor x
        /// @param x Tensor x
        /// @param y Tensor y
        template <typename T, typename U>
        void AddVectors(const double a, const Tensor<T> &x, Tensor<U> &y);

        /// @brief Copies elements from one tensor to another, Mixed precision version
        /// @brief if U == T, wrapper to the single-precision or double-precision version
        /// @tparam T Numerical type of the source tensor
        /// @tparam U Numerical type of the destination tensor
        /// @param X Source tensor
        /// @param Y Destination tensor
        template <typename T, typename U>
        void CopyVectors(const Tensor<T> &X, Tensor<U> &Y);

        /// @brief Copies elements from one tensor to another with strides, wrapper to BLAS function <T>copy
        /// @tparam T Numerical type of the tensors
        /// @param X Source tensor
        /// @param strideX Stride between elements in tensor X
        /// @param Y Destination tensor
        /// @param strideY Stride between elements in tensor Y
        template <typename T>
        void CopyVectors(const Tensor<T> &X, const size_t strideX, Tensor<T> &Y, const size_t strideY);

        /// @brief Copies elements from one tensor to another, wrapper to BLAS function <T>swap, unit stride version
        /// @tparam T Numerical type of the tensors
        /// @param X Tensor X
        /// @param Y Tensor Y
        template <typename T>
        void SwapVectors(Tensor<T> &X, Tensor<T> &Y);

        /// @brief Copies elements from one tensor to another with strides, wrapper to BLAS function <T>swap
        /// @tparam T Numerical type of the tensors
        /// @param X Tensor X
        /// @param strideX Stride between elements in tensor X
        /// @param Y Tensor Y
        /// @param strideY Stride between elements in tensor Y
        template <typename T>
        void SwapVectors(Tensor<T> &X, const size_t strideX, Tensor<T> &Y, const size_t strideY);

        /// @brief Scales a tensor treated as vector by a scalar multiplier, wrapper to BLAS function <T>scal
        /// @tparam T Numerical type of the tensor
        /// @param a scalar multiplier
        /// @param x Tensor x
        template <typename T>
        void ScaleVector(const T a, Tensor<T> &x);

        /// @brief Scales a tensor treated as vector by a scalar multiplier, wrapper to BLAS function <T>scal
        /// @tparam Scalar numerical type of the scalar multiplier
        /// @tparam T numerical type of the tensor
        /// @param a scalar multiplier
        /// @param x Tensor x
        template <typename Scalar, typename T>
        void ScaleVector(const Scalar a, Tensor<T> &x)
        {
            ScaleVector((T)a, x);
        };

        /// @brief Scales a tensor treated as vector by a scalar multiplier with stride, wrapper to BLAS function <T>scal
        /// @tparam T Numerical type of the tensor
        /// @param a scalar multiplier
        /// @param x Tensor x
        /// @param ix Stride between elements in tensor x
        template <typename T>
        void ScaleVector(const T a, Tensor<T> &x, size_t ix);

        /// @brief Scales a tensor treated as vector by a scalar multiplier with stride, wrapper to BLAS function <T>scal
        /// @tparam Scalar numerical type of the scalar multiplier
        /// @tparam T numerical type of the tensor
        /// @param a scalar multiplier
        /// @param x Tensor x
        /// @param ix Stride between elements in tensor x
        template <typename Scalar, typename T>
        void ScaleVector(const Scalar a, Tensor<T> &x, size_t ix)
        {
            ScaleVector((T)a, x, ix);
        };

        /// @brief Finds the index of the maximum element in a tensor treated as vector, wrapper to BLAS function <T>iamax
        /// @tparam T Numerical type of the tensor
        /// @param x Tensor x
        template <typename T>
        int IndexMaxFromVector(const Tensor<T> &x);

        /// @brief Finds the index of the maximum element in a tensor treated as vector with stride, wrapper to BLAS function <T>iamax
        /// @tparam T Numerical type of the tensor
        /// @param x Tensor x
        /// @param ix Stride between elements in tensor x
        template <typename T>
        int IndexMaxFromVector(const Tensor<T> &x, const size_t ix);

        /// @brief Wrapper function to AddVectors with CPURuntime as first argument
        /// @tparam ...Args allowing for any number of arguments
        /// @param rt_ CPU Runtime instance
        /// @param ...args remaining arguments forwarded to AddVectors
        template <typename... Args>
        void AddVectors(const CPURuntime &rt_, Args &&...args)
        {
            (AddVectors(args...));
        }

        /// @brief Wrapper function to CopyVectors with CPURuntime as first argument
        /// @tparam ...Args allowing for any number of arguments
        /// @param rt_ CPU Runtime instance
        /// @param ...args remaining arguments forwarded to CopyVectors
        template <typename... Args>
        void CopyVectors(const CPURuntime &rt_, Args &&...args)
        {
            (CopyVectors(args...));
        }

        /// @brief Wrapper function to SwapVectors with CPURuntime as first argument
        /// @tparam ...Args allowing for any number of arguments
        /// @param rt_ CPU Runtime instance
        /// @param ...args remaining arguments forwarded to SwapVectors
        template <typename... Args>
        void SwapVectors(const CPURuntime &rt_, Args &&...args)
        {
            (SwapVectors(args...));
        }

        /// @brief Wrapper function to ScaleVector with CPURuntime as first argument
        /// @tparam ...Args allowing for any number of arguments
        /// @param rt_ CPU Runtime instance
        /// @param ...args remaining arguments forwarded to ScaleVector
        template <typename... Args>
        void ScaleVector(const CPURuntime &rt_, Args &&...args)
        {
            (ScaleVector(args...));
        }

        /// @brief Wrapper function to IndexMaxFromVector with CPURuntime as first argument
        /// @tparam ...Args allowing for any number of arguments
        /// @param rt_ CPU Runtime instance
        /// @param ...args remaining arguments forwarded to IndexMaxFromVector
        /// @return index of the maximum element in the vector
        template <typename... Args>
        int IndexMaxFromVector(const CPURuntime &rt_, Args &&...args)
        {
            return (IndexMaxFromVector(args...));
        }
    }
}

#endif