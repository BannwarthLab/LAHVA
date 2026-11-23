/// @file unpack-vector.cu
/// @brief GPU kernels for packing and unpacking block vectors between scattered and padded layout.
///
/// Provides CUDA kernel implementations for gathering vector data into a padded block layout
/// before batched operations, and for scattering packed block data back to the original
/// vector layout afterwards.

#include "runtime.hpp"

namespace lahva
{
    namespace gpu
    {
        /// @brief Kernel grid that covers every element of every matrix block.
        ///
        /// The x dimension selects the matrix block. A matrix block can have more elements than a
        /// kernel block has threads, so the y dimension selects a chunk of that matrix block.
        ///
        /// @param cudart CUDA runtime context for the block size configuration.
        /// @param max_size Maximum size of any block (padding size).
        /// @param num_blocks Number of blocks.
        /// @return Grid dimensions for launching a pack or unpack kernel.
        static dim3 block_chunk_grid(const CudaRuntime &cudart, size_t max_size, int num_blocks)
        {
            const size_t threads_per_block = cudart.blockSize();
            const size_t chunks_per_block = std::max<size_t>((max_size + threads_per_block - 1) / threads_per_block, 1);
            return dim3(static_cast<unsigned int>(num_blocks), static_cast<unsigned int>(chunks_per_block));
        }

        /// @brief GPU kernel for packing block vector from scattered to padded layout.
        ///
        /// Gathers elements from the original vector layout into a padded block layout.
        /// Each block is stored contiguously in padded format with fixed size `max_size`.
        /// Padding elements of dst are left untouched.
        ///
        /// @tparam T Floating point type (float or double).
        /// @param src Input vector in scattered layout [total_size].
        /// @param dst Output vector in padded block layout [num_blocks * max_size].
        /// @param offsets Device array of starting positions for each block in input [num_blocks].
        /// @param block_sizes Device array of actual sizes for each block [num_blocks].
        /// @param max_size Maximum size of any block (padding size).
        /// @param num_blocks Number of blocks.
        template <typename T>
        __global__ void pack_vector_kernel(
            const T *src,
            T *dst,
            const int *offsets,
            const int *block_sizes,
            size_t max_size,
            int num_blocks)
        {
            int block_idx = blockIdx.x;
            size_t element_idx = blockIdx.y * blockDim.x + threadIdx.x;

            if (block_idx >= num_blocks) return;

            size_t block_size = block_sizes[block_idx];
            if (element_idx >= block_size) return;

            int offset = offsets[block_idx];
            dst[block_idx * max_size + element_idx] = src[offset + element_idx];
        }

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
            size_t element_idx = blockIdx.y * blockDim.x + threadIdx.x;

            if (block_idx >= num_blocks) return;

            size_t block_size = block_sizes[block_idx];
            if (element_idx >= block_size) return;

            int offset = offsets[block_idx];
            dst[offset + element_idx] = src[block_idx * max_size + element_idx];
        }

        /// @brief Packs block vector from scattered to padded layout on device.
        ///
        /// Launches GPU kernel to gather a vector into the padded block layout used by batched
        /// block operations. Reads the device copy of the vector, which holds the results of
        /// earlier GPU operations.
        ///
        /// @tparam T Floating point type (float or double).
        /// @param cudart CUDA runtime context for stream and block size configuration.
        /// @param src Device pointer to input vector in scattered layout [total_size].
        /// @param dst Device pointer to padded block vector [num_blocks * max_size].
        /// @param d_offsets Device array of starting positions for each block [num_blocks].
        /// @param d_block_sizes Device array of actual block sizes [num_blocks].
        /// @param max_size Maximum block size (padding size).
        /// @param num_blocks Number of blocks.
        template <typename T>
        void pack_vector(
            const CudaRuntime &cudart,
            const T *src,
            T *dst,
            const int *d_offsets,
            const int *d_block_sizes,
            size_t max_size,
            int num_blocks)
        {
            pack_vector_kernel<<<block_chunk_grid(cudart, max_size, num_blocks), cudart.blockSize(), 0, cudart.getStream()>>>(
                src, dst, d_offsets, d_block_sizes, max_size, num_blocks);
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
            unpack_vector_kernel<<<block_chunk_grid(cudart, max_size, num_blocks), cudart.blockSize(), 0, cudart.getStream()>>>(
                src, dst, d_offsets, d_block_sizes, max_size, num_blocks);
        }

        // Explicit template instantiation for the kernels
        template __global__ void pack_vector_kernel<float>(
            const float *src, float *dst, const int *offsets, const int *block_sizes,
            size_t max_size, int num_blocks);
        template __global__ void pack_vector_kernel<double>(
            const double *src, double *dst, const int *offsets, const int *block_sizes,
            size_t max_size, int num_blocks);
        template __global__ void unpack_vector_kernel<float>(
            const float *src, float *dst, const int *offsets, const int *block_sizes,
            size_t max_size, int num_blocks);
        template __global__ void unpack_vector_kernel<double>(
            const double *src, double *dst, const int *offsets, const int *block_sizes,
            size_t max_size, int num_blocks);

        // Explicit template instantiation for pack_vector and unpack_vector
        template void pack_vector<float>(
            const CudaRuntime &cudart,
            const float *src, float *dst, const int *d_offsets, const int *d_block_sizes,
            size_t max_size, int num_blocks);
        template void pack_vector<double>(
            const CudaRuntime &cudart,
            const double *src, double *dst, const int *d_offsets, const int *d_block_sizes,
            size_t max_size, int num_blocks);
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
