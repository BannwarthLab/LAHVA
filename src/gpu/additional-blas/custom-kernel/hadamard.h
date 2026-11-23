/// @file hadamard.h
/// @brief GPU kernels for Hadamard product (element-wise multiplication) operations.
///
/// Provides custom GPU kernels for efficient element-wise tensor multiplication
/// in both in-place and out-of-place variants with single and double precision.

#pragma once
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "runtime.hpp"

namespace lahva
{
    namespace gpu
    {
        /// @brief In-place Hadamard product for single-precision vectors: vecinout *= vecin.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param vecinout Input/output single-precision vector.
        /// @param vecin Input single-precision vector.
        /// @param ndim2 Number of elements.
        void sHadamard(const CudaRuntime& cudart, float* vecinout, const float* vecin, size_t ndim2);

        /// @brief In-place Hadamard product for double-precision vectors: vecinout *= vecin.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param vecinout Input/output double-precision vector.
        /// @param vecin Input double-precision vector.
        /// @param ndim2 Number of elements.
        void dHadamard(const CudaRuntime& cudart, double* vecinout, const double* vecin, size_t ndim2);

        /// @brief Hadamard product with separate output for single-precision vectors: vecout = vecin1 * vecin2.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param vecout Output single-precision vector.
        /// @param vecin1 First input single-precision vector.
        /// @param vecin2 Second input single-precision vector.
        /// @param ndim2 Number of elements.
        void sHadamard(const CudaRuntime& cudart, float* vecout, const float* vecin1, const float* vecin2, size_t ndim2);

        /// @brief Hadamard product with separate output for double-precision vectors: vecout = vecin1 * vecin2.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param vecout Output double-precision vector.
        /// @param vecin1 First input double-precision vector.
        /// @param vecin2 Second input double-precision vector.
        /// @param ndim2 Number of elements.
        void dHadamard(const CudaRuntime& cudart, double* vecout, const double* vecin1, const double* vecin2, size_t ndim2);

    } // namespace gpu

} // namespace lahva
