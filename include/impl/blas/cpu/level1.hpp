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
    double InnerVectorProduct(const Tensor_<double>& X, const Tensor_<double>& Y);
    float InnerVectorProduct(const Tensor_<float>& X, const Tensor_<float>& Y);
    double InnerVectorProduct(const Tensor_<double>& X, const size_t strideX, const Tensor_<double>& Y, const size_t strideY);
    float InnerVectorProduct(const Tensor_<float>& X, const size_t strideX, const Tensor_<float>& Y, const size_t strideY);

    double InnerVectorProduct(const CPURuntime& rt_, const Tensor_<double>& X, const Tensor_<double>& Y);
    float InnerVectorProduct(const CPURuntime& rt_, const Tensor_<float>& X, const Tensor_<float>& Y);
    double InnerVectorProduct(const CPURuntime& rt_, const Tensor_<double>& X, const size_t strideX, const Tensor_<double>& Y, const size_t strideY);
    float InnerVectorProduct(const CPURuntime& rt_, const Tensor_<float>& X, const size_t strideX, const Tensor_<float>& Y, const size_t strideY);

    void AddVectors(const double a, const Tensor_<double>& x, Tensor_<double>& y);
    void AddVectors(const float a, const Tensor_<float>& x, Tensor_<float>& y);
    void AddVectors(const double a, const Tensor_<double>& x, size_t ix, Tensor_<double>& y, size_t iy);
    void AddVectors(const float a, const Tensor_<float>& x, size_t ix, Tensor_<float>& y, size_t iy);
    void AddVectors(const double a, const Tensor_<double>& x, Tensor_<float>& y);
    void AddVectors(const double a, const Tensor_<float>& x, Tensor_<double>& y);

    void CopyVectors(const Tensor_<double>& X, Tensor_<double>& Y);
    void CopyVectors(const Tensor_<float>& X, Tensor_<float>& Y);
    void CopyVectors(const Tensor_<double>& X, const size_t strideX, Tensor_<double>& Y, const size_t strideY);
    void CopyVectors(const Tensor_<float>& X, const size_t strideX, Tensor_<float>& Y, const size_t strideY);
    void CopyVectors(const Tensor_<float>& X, Tensor_<double>& Y);
    void CopyVectors(const Tensor_<double>& X, Tensor_<float>& Y);

    void SwapVectors(Tensor_<double>& X, Tensor_<double>& Y);
    void SwapVectors(Tensor_<float>& X, Tensor_<float>& Y);
    void SwapVectors(Tensor_<double>& X, const size_t strideX, Tensor_<double>& Y, const size_t strideY);
    void SwapVectors(Tensor_<float>& X, const size_t strideX, Tensor_<float>& Y, const size_t strideY);

    void ScaleVector(const double a, Tensor_<double>& x);
    void ScaleVector(const float a, Tensor_<float>& x);
    void ScaleVector(const double a, Tensor_<double>& x, size_t ix);
    void ScaleVector(const float a, Tensor_<float>& x, size_t ix);

    int IndexMaxFromVector(const Tensor_<double>& x);
    int IndexMaxFromVector(const Tensor_<float>& x);
    int IndexMaxFromVector(const Tensor_<double>& x, const size_t ix);
    int IndexMaxFromVector(const Tensor_<float>& x, const size_t ix);

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
