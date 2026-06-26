/// @file additional-level1.hpp
/// @brief GPU kernel declarations for additional Level-1 operations.
///
/// Provides GPU kernel interface for mixed-precision tensor operations
/// and element-wise transformations.

#pragma once
#include <cuda_runtime.h>
namespace lahva
{
    namespace gpu
    {
        /// @brief GPU kernel for copying and converting tensors between types.
        ///
        /// Performs element-wise copy with type conversion from input to output tensor.
        /// Supports mixed-precision conversions (e.g., double to float, float to double).
        ///
        /// @tparam in Input element type.
        /// @tparam out Output element type.
        /// @param size Number of elements to copy.
        /// @param d_in Input device pointer.
        /// @param d_out Output device pointer.
        template<typename in, typename out>
        __global__ void CopyTensors_(unsigned long size, const in* d_in, out* d_out);

        /// @brief Host wrapper for tensor copy and conversion kernel.
        ///
        /// Launches the GPU kernel to copy and convert tensors.
        ///
        /// @tparam in Input element type.
        /// @tparam out Output element type.
        /// @param size Number of elements to copy.
        /// @param d_in Input device pointer.
        /// @param d_out Output device pointer.
        template<typename in, typename out>
        void CopyTensors(const unsigned long size, const in* d_in, out* d_out);

    } // namespace gpu
} // namespace lahva
