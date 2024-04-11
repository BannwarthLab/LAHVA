#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <math.h>
#include "hadamard.h"
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

void sHadamard(float* vecinout, const float* vecin, size_t ndim2){
    int blockSize, gridSize;
    blockSize = 512;
    // Number of thread blocks in grid
    gridSize = (int)ceil((float)ndim2/blockSize);

    sElementwiseScale<<<gridSize,blockSize>>>(vecinout, vecin, ndim2);
}

void dHadamard(double* vecinout, const double* vecin, size_t ndim2){
    int blockSize, gridSize;
    blockSize = 512;
    // Number of thread blocks in grid
    gridSize = (int)ceil((float)ndim2/blockSize);

    dElementwiseScale<<<gridSize,blockSize>>>(vecinout, vecin, ndim2);
}

void sHadamardcopy(float* vecout, const float* vecin1, const float* vecin2, size_t ndim2){
    int blockSize, gridSize;
    blockSize = 512;
    // Number of thread blocks in grid
    gridSize = (int)ceil((float)ndim2/blockSize);

    sElementwiseScalecopy<<<gridSize,blockSize>>>(vecout, vecin1, vecin2, ndim2);
}

void dHadamardcopy(double* vecout, const double* vecin1, const double* vecin2, size_t ndim2){
    int blockSize, gridSize;
    blockSize = 512;
    // Number of thread blocks in grid
    gridSize = (int)ceil((float)ndim2/blockSize);

    dElementwiseScalecopy<<<gridSize,blockSize>>>(vecout, vecin1, vecin2, ndim2);
}

  }
}