/// @file unpack-vector.hpp
/// @brief GPU kernel declarations for unpacking block vectors.

#pragma once

namespace lahva
{
    // Forward declarations
    class CudaRuntime;
    template <typename T>
    class Vector;

    namespace gpu
    {
        /// @brief Unpacks block vector from padded to scattered layout on device.
        ///
        /// Launches GPU kernel to redistribute padded block vector back to original layout.
        /// Used after batched block operations to restore vector layout for further computation.
        /// Each block is stored contiguously in padded format with fixed size `max_size` on the GPU,
        /// and is scattered back to its original position based on offset and block size information.
        ///
        /// @tparam T Floating point type (float or double).
        /// @param[in] cudart CUDA runtime context for stream and block size configuration.
        /// @param[in] src Device pointer to padded block vector [num_blocks * max_size].
        /// @param[out] dst Device pointer to output vector in scattered layout [total_size].
        /// @param[in] d_offsets Device array of starting positions for each block [num_blocks].
        /// @param[in] d_block_sizes Device array of actual block sizes [num_blocks].
        /// @param[in] max_size Maximum block size (padding size for uniform blocks).
        /// @param[in] num_blocks Number of blocks to unpack.
        template <typename T>
        void unpack_vector(
            const CudaRuntime &cudart,
            const T *src,
            T *dst,
            const int *d_offsets,
            const int *d_block_sizes,
            size_t max_size,
            int num_blocks);


    } // namespace gpu
} // namespace lahva
