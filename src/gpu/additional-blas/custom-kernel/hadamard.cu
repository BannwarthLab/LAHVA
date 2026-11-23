/// @file hadamard.cu
/// @brief GPU kernel implementations for Hadamard product (element-wise multiplication).
///
/// Provides CUDA kernel implementations for efficient element-wise multiplication
/// operations on GPU vectors with single and double precision support.

#include <math.h>
#include "hadamard.h"

namespace lahva
{
    namespace gpu
    {
        /// @brief GPU kernel for in-place double-precision Hadamard product.
        ///
        /// Computes element-wise multiplication: vecinout[i] *= vecin[i] in parallel on GPU.
        ///
        /// @param vecinout Input/output double-precision vector.
        /// @param vecin Input double-precision vector.
        /// @param ndim2 Number of elements.
        __global__ static void dElementwiseScale(double *vecinout, const double *vecin, size_t ndim2)
        {
            // get the global id (in vector)
            size_t id = blockIdx.x * blockDim.x + threadIdx.x;
            if (id < ndim2)
                vecinout[id] *= vecin[id];
        }

        /// @brief GPU kernel for in-place single-precision Hadamard product.
        ///
        /// Computes element-wise multiplication: vecinout[i] *= vecin[i] in parallel on GPU.
        ///
        /// @param vecinout Input/output single-precision vector.
        /// @param vecin Input single-precision vector.
        /// @param ndim2 Number of elements.
        __global__ static void sElementwiseScale(float *vecinout, const float *vecin, size_t ndim2)
        {
            // get the global id (in vector)
            size_t id = blockIdx.x * blockDim.x + threadIdx.x;
            if (id < ndim2)
                vecinout[id] *= vecin[id];
        }

        /// @brief GPU kernel for Hadamard product with separate output for double-precision.
        ///
        /// Computes element-wise multiplication with separate output: vecout[i] = vecin1[i] * vecin2[i].
        ///
        /// @param vecout Output double-precision vector.
        /// @param vecin1 First input double-precision vector.
        /// @param vecin2 Second input double-precision vector.
        /// @param ndim2 Number of elements.
        __global__ static void dElementwiseScale(double *vecout, const double *vecin1, const double *vecin2, size_t ndim2)
        {
            // get the global id (in vector)
            size_t id = blockIdx.x * blockDim.x + threadIdx.x;
            if (id < ndim2)
                vecout[id] = vecin1[id] * vecin2[id];
        }

        /// @brief GPU kernel for Hadamard product with separate output for single-precision.
        ///
        /// Computes element-wise multiplication with separate output: vecout[i] = vecin1[i] * vecin2[i].
        ///
        /// @param vecout Output single-precision vector.
        /// @param vecin1 First input single-precision vector.
        /// @param vecin2 Second input single-precision vector.
        /// @param ndim2 Number of elements.
        __global__ static void sElementwiseScale(float *vecout, const float *vecin1, const float *vecin2, size_t ndim2)
        {
            // get the global id (in vector)
            size_t id = blockIdx.x * blockDim.x + threadIdx.x;
            if (id < ndim2)
                vecout[id] = vecin1[id] * vecin2[id];
        }

        /// @brief GPU kernel for in-place single-precision Hadamard product.
        ///
        /// Computes element-wise multiplication: vecinout[i] *= vecin[i] in parallel on GPU.
        ///
        /// @param vecinout Input/output single-precision vector.
        /// @param vecin Input single-precision vector.
        /// @param ndim2 Number of elements.
        void sHadamard(const CudaRuntime &cudart, float *vecinout, const float *vecin, size_t ndim2)
        {
            sElementwiseScale<<<cudart.gridSize(ndim2, 1), cudart.blockSize(), 0, cudart.getStream()>>>(vecinout, vecin, ndim2);
        }

    
        /// @brief GPU kernel for in-place double-precision Hadamard product.
        ///
        /// Computes element-wise multiplication: vecinout[i] *= vecin[i] in parallel on GPU.
        ///
        /// @param vecinout Input/output double-precision vector.
        /// @param vecin Input double-precision vector.
        /// @param ndim2 Number of elements.
        void dHadamard(const CudaRuntime &cudart, double *vecinout, const double *vecin, size_t ndim2)
        {
            dElementwiseScale<<<cudart.gridSize(ndim2, 1), cudart.blockSize(), 0, cudart.getStream()>>>(vecinout, vecin, ndim2);
        }
        /// @brief GPU kernel for Hadamard product with separate output for single-precision.
        ///
        /// Computes element-wise multiplication with separate output: vecout[i] = vecin1[i] * vecin2[i].
        ///
        /// @param vecout Output single-precision vector.
        /// @param vecin1 First input single-precision vector.
        /// @param vecin2 Second input single-precision vector.
        /// @param ndim2 Number of elements.
        void sHadamard(const CudaRuntime &cudart, float *vecout, const float *vecin1, const float *vecin2, size_t ndim2)
        {
            sElementwiseScale<<<cudart.gridSize(ndim2, 1), cudart.blockSize(), 0, cudart.getStream()>>>(vecout, vecin1, vecin2, ndim2);
        }

        /// @brief GPU kernel for Hadamard product with separate output for double-precision.
        ///
        /// Computes element-wise multiplication with separate output: vecout[i] = vecin1[i] * vecin2[i].
        ///
        /// @param vecout Output double-precision vector.
        /// @param vecin1 First input double-precision vector.
        /// @param vecin2 Second input double-precision vector.
        /// @param ndim2 Number of elements.
        void dHadamard(const CudaRuntime &cudart, double *vecout, const double *vecin1, const double *vecin2, size_t ndim2)
        {
            dElementwiseScale<<<cudart.gridSize(ndim2, 1), cudart.blockSize(), 0, cudart.getStream()>>>(vecout, vecin1, vecin2, ndim2);
        }

    }
}