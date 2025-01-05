#include <math.h>
#include "hadamard.h"

namespace lahva
{
    namespace gpu
    {

        // this scales the element of the first vector by the element of the second vector
        __global__ static void dElementwiseScale(double *vecinout, const double *vecin, size_t ndim2)
        {
            // get the global id (in vector)
            size_t id = blockIdx.x * blockDim.x + threadIdx.x;
            if (id < ndim2)
                vecinout[id] *= vecin[id];
        }

        // this scales the element of the first vector by the element of the second vector
        __global__ static void sElementwiseScale(float *vecinout, const float *vecin, size_t ndim2)
        {
            // get the global id (in vector)
            size_t id = blockIdx.x * blockDim.x + threadIdx.x;
            if (id < ndim2)
                vecinout[id] *= vecin[id];
        }

        __global__ static void dElementwiseScalecopy(double *vecout, const double *vecin1, const double *vecin2, size_t ndim2)
        {
            // get the global id (in vector)
            size_t id = blockIdx.x * blockDim.x + threadIdx.x;
            if (id < ndim2)
                vecout[id] = vecin1[id] * vecin2[id];
        }

        // this scales the element of the first vector by the element of the second vector
        __global__ static void sElementwiseScalecopy(float *vecout, const float *vecin1, const float *vecin2, size_t ndim2)
        {
            // get the global id (in vector)
            size_t id = blockIdx.x * blockDim.x + threadIdx.x;
            if (id < ndim2)
                vecout[id] = vecin1[id] * vecin2[id];
        }

        __global__ static void dElementwiseScalecopyMV(double *vecout, const double *vecin1, const double *vecin2, size_t ndim2, size_t ndim)
        {
            // get the global id (in vector)
            unsigned long long id = blockIdx.y;
            unsigned long long jd = threadIdx.x + blockIdx.x * blockDim.x;
            if (id < ndim && jd < ndim)
            {
                vecout[id* ndim + jd] = vecin1[id* ndim+ jd] * vecin2[id];
            }
                
        }

        // this scales the element of the first vector by the element of the second vector
        __global__ static void sElementwiseScalecopyMV(float *vecout, const float *vecin1, const float *vecin2, size_t ndim2, size_t ndim)
        {
            // get the global id (in vector)
            size_t id = blockIdx.x * blockDim.x + threadIdx.x;
            if (id < ndim2)
                vecout[id] = vecin1[id] * vecin2[id % ndim];
        }

        void sHadamard(const CudaRuntime &cudart, float *vecinout, const float *vecin, size_t ndim2)
        {

            sElementwiseScale<<<cudart.gridSize(ndim2, 1), cudart.blockSize(), 0, cudart.getStream()>>>(vecinout, vecin, ndim2);
        }

        void dHadamard(const CudaRuntime &cudart, double *vecinout, const double *vecin, size_t ndim2)
        {

            dElementwiseScale<<<cudart.gridSize(ndim2, 1), cudart.blockSize(), 0, cudart.getStream()>>>(vecinout, vecin, ndim2);
        }

        void sHadamardcopy(const CudaRuntime &cudart, float *vecout, const float *vecin1, const float *vecin2, size_t ndim2)
        {

            sElementwiseScalecopy<<<cudart.gridSize(ndim2, 1), cudart.blockSize(), 0, cudart.getStream()>>>(vecout, vecin1, vecin2, ndim2);
        }

        void dHadamardcopy(const CudaRuntime &cudart, double *vecout, const double *vecin1, const double *vecin2, size_t ndim2)
        {

            dElementwiseScalecopy<<<cudart.gridSize(ndim2, 1), cudart.blockSize(), 0, cudart.getStream()>>>(vecout, vecin1, vecin2, ndim2);
        }

        void sHadamardcopy(const CudaRuntime &cudart, float *vecout, const float *vecin1, const float *vecin2, size_t ndim2, size_t ndim)
        {

            sElementwiseScalecopyMV<<<cudart.gridSize(ndim2, 1), cudart.blockSize(), 0, cudart.getStream()>>>(vecout, vecin1, vecin2, ndim2, ndim);
        }

        void dHadamardcopy(const CudaRuntime &cudart, double *vecout, const double *vecin1, const double *vecin2, size_t ndim2, size_t ndim)
        {
            dim3 nThreadsPerBlock(cudart.blockSize(), 1);
            dim3 nBlocksPerGrid(ceil((double)(ndim) / double(cudart.blockSize())), ndim);
            dElementwiseScalecopyMV<<<nBlocksPerGrid, nThreadsPerBlock, 0, cudart.getStream()>>>(vecout, vecin1, vecin2, ndim2, ndim);
        }

    }
}