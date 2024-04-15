#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <math.h>
#include "hadamard.h"
#include "runtime.hpp"
namespace tcgmtensor{
  namespace gpu{

// this scales the element of the first vector by the element of the second vector 
__global__ static void dElementwiseScale(double* vecinout, const double* vecin,size_t ndim2)
{
  // get the global id (in vector)
  size_t id = blockIdx.x*blockDim.x+threadIdx.x; 
  if (id < ndim2) vecinout[id] *= vecin[id];
}

// this scales the element of the first vector by the element of the second vector 
__global__ static void sElementwiseScale(float* vecinout, const float* vecin,size_t ndim2)
{
  // get the global id (in vector)
  size_t id = blockIdx.x*blockDim.x+threadIdx.x; 
  if (id < ndim2) vecinout[id] *= vecin[id];
}

__global__ static void dElementwiseScalecopy(double* vecout, const double* vecin1, const double* vecin2, size_t ndim2)
{
  // get the global id (in vector)
  size_t id = blockIdx.x*blockDim.x+threadIdx.x; 
  if (id < ndim2) vecout[id] = vecin1[id] * vecin2[id];
}

// this scales the element of the first vector by the element of the second vector 
__global__ static void sElementwiseScalecopy(float* vecout, const float* vecin1, const float* vecin2, size_t ndim2)
{
  // get the global id (in vector)
  size_t id = blockIdx.x*blockDim.x+threadIdx.x; 
  if (id < ndim2) vecout[id] = vecin1[id] * vecin2[id];
}

void sHadamard(const CudaRuntime& cudart, float* vecinout, const float* vecin, size_t ndim2){

    sElementwiseScale<<<cudart.gridSize(ndim2, 1),cudart.blockSize()>>>(vecinout, vecin, ndim2);
}

void dHadamard(const CudaRuntime& cudart, double* vecinout, const double* vecin, size_t ndim2){

    dElementwiseScale<<<cudart.gridSize(ndim2, 1),cudart.blockSize()>>>(vecinout, vecin, ndim2);
}

void sHadamardcopy(const CudaRuntime& cudart, float* vecout, const float* vecin1, const float* vecin2, size_t ndim2){

    sElementwiseScalecopy<<<cudart.gridSize(ndim2, 1),cudart.blockSize()>>>(vecout, vecin1, vecin2, ndim2);
}

void dHadamardcopy(const CudaRuntime& cudart, double* vecout, const double* vecin1, const double* vecin2, size_t ndim2){

    dElementwiseScalecopy<<<cudart.gridSize(ndim2, 1),cudart.blockSize()>>>(vecout, vecin1, vecin2, ndim2);
}

  }
}