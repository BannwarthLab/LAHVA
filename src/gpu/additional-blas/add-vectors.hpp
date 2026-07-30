/// @file add-vectors.hpp
/// @brief Mixed-precision vector addition kernels for GPU computation.
///
/// Provides functions for adding vectors with type conversion between different
/// precisions (double, float, half-precision).

#pragma once
#include "linalg.hpp"

namespace lahva
{
    namespace gpu
    {
        /// @brief Adds double-precision vector to single-precision vector: y = a*x + y.
        ///
        /// Performs mixed-precision scaled vector addition converting from double to float.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param ndim Number of elements in vectors.
        /// @param a Scalar factor for vector x.
        /// @param x Input double-precision tensor.
        /// @param y Input/output single-precision tensor.
        void AddVector(const CudaRuntime& cudart, unsigned long long ndim, const double a, const Tensor_<double>& x, Tensor_<float>& y);

        /// @brief Adds single-precision vector to double-precision vector: y = a*x + y.
        ///
        /// Performs mixed-precision scaled vector addition converting from float to double.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param ndim Number of elements in vectors.
        /// @param a Scalar factor for vector x.
        /// @param x Input single-precision tensor.
        /// @param y Input/output double-precision tensor.
        void AddVector(const CudaRuntime& cudart, unsigned long long ndim, const double a, const Tensor_<float>& x, Tensor_<double>& y);

        /// @brief Adds half-precision vector to double-precision vector: y = a*x + y.
        ///
        /// Performs mixed-precision scaled vector addition converting from FP16 to double.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param ndim Number of elements in vectors.
        /// @param a Scalar factor for vector x.
        /// @param x Input half-precision tensor.
        /// @param y Input/output double-precision tensor.
        void AddVector(const CudaRuntime& cudart, unsigned long long ndim, const double a, const Tensor_<__half>& x, Tensor_<double>& y);
    } // namespace gpu

} // namespace lahva
