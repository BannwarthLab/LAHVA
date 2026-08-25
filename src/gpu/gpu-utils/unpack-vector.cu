/// @file unpack-vector.cu
/// @brief GPU kernel for unpacking block vector from padded to scattered layout.
///
/// Provides CUDA kernel implementation for scattering packed block data back
/// to the original vector layout after batched operations.

#include "runtime.hpp"

namespace lahva
{
    namespace gpu
    {
        /// @brief GPU kernel for unpacking block vector from padded to scattered layout.
        ///
        /// Scatters elements from a padded block layout back to the original vector layout.
        /// Each block is stored contiguously in padded format with fixed size `max_size`.
        /// This kernel redistributes blocks to their original positions in the output vector.
        ///
        /// @tparam T Floating point type (float or double).
        /// @param src Input vector in padded block layout [num_blocks * max_size].
        /// @param dst Output vector in scattered layout [total_size].
        /// @param offsets Device array of starting positions for each block in output [num_blocks].
        /// @param block_sizes Device array of actual sizes for each block [num_blocks].
        /// @param max_size Maximum size of any block (padding size).
        /// @param num_blocks Number of blocks.
        template <typename T>
        __global__ void unpack_vector_kernel(
            const T *src,
            T *dst,
            const int *offsets,
            const int *block_sizes,
            size_t max_size,
            int num_blocks)
        {
            int block_idx = blockIdx.x;
            int thread_idx = threadIdx.x;

            if (block_idx >= num_blocks) return;

            size_t block_size = block_sizes[block_idx];
            if (thread_idx >= block_size) return;

            int offset = offsets[block_idx];
            dst[offset + thread_idx] = src[block_idx * max_size + thread_idx];
        }


        /// @brief Unpacks block vector from padded to scattered layout on device.
        ///
        /// Launches GPU kernel to redistribute padded block vector back to original layout.
        /// Used after batched block operations to restore vector layout for further computation.
        ///
        /// @tparam T Floating point type (float or double).
        /// @param cudart CUDA runtime context for stream and block size configuration.
        /// @param src Device pointer to padded block vector [num_blocks * max_size].
        /// @param dst Device pointer to output vector in scattered layout [total_size].
        /// @param d_offsets Device array of starting positions for each block [num_blocks].
        /// @param d_block_sizes Device array of actual block sizes [num_blocks].
        /// @param max_size Maximum block size (padding size).
        /// @param num_blocks Number of blocks.
        template <typename T>
        void unpack_vector(
            const CudaRuntime &cudart,
            const T *src,
            T *dst,
            const int *d_offsets,
            const int *d_block_sizes,
            size_t max_size,
            int num_blocks)
        {
            unpack_vector_kernel<<<num_blocks, cudart.blockSize(), 0, cudart.getStream()>>>(
                src, dst, d_offsets, d_block_sizes, max_size, num_blocks);
        }

        // Explicit template instantiation for the kernel
        template __global__ void unpack_vector_kernel<float>(
            const float *src, float *dst, const int *offsets, const int *block_sizes,
            size_t max_size, int num_blocks);
        template __global__ void unpack_vector_kernel<double>(
            const double *src, double *dst, const int *offsets, const int *block_sizes,
            size_t max_size, int num_blocks);

        // Explicit template instantiation for unpack_vector
        template void unpack_vector<float>(
            const CudaRuntime &cudart,
            const float *src, float *dst, const int *d_offsets, const int *d_block_sizes,
            size_t max_size, int num_blocks);
        template void unpack_vector<double>(
            const CudaRuntime &cudart,
            const double *src, double *dst, const int *d_offsets, const int *d_block_sizes,
            size_t max_size, int num_blocks);

    } // namespace gpu
} // namespace lahva
