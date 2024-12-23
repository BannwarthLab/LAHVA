#include "impl/blas/gpu/additional-level1.hpp"
#include "../../../gpu-utils/utils.hpp"

namespace tcgmtensor
{
    namespace gpu
    {
        __global__ static void Frobenius(const unsigned long long size, const float* mat, float* sum)
        {
            unsigned long long tid = threadIdx.x;
            unsigned long long id = blockIdx.x * blockDim.x + threadIdx.x;
            extern __shared__ float temp[THREADS_PER_BLOCK];
            //due to the shared directive every block gets its own copy of tmp
            float tmp = 0.0;
            while (id < size)
            {
                tmp += mat[id]*mat[id];
                id += blockDim.x * gridDim.x;
            }
            temp[tid] = 0;
            temp[tid] = tmp;
            __syncthreads();

            // for reductions, threadsPerBlock must be a power of 2
            // because of the following code
            int i = blockDim.x / 2;
            while (i != 0)
            {
                if (tid < i)

                    temp[tid] += temp[tid + i];

                __syncthreads();

                i /= 2;
            }
            // Thread 0 adds partial sums to overall sum
            if (tid == 0)
                sum[blockIdx.x] = temp[0]; 
        }

        __global__ static void Frobenius(const unsigned long long size, const double* mat, double* sum)
        {
            unsigned long long tid = threadIdx.x;
            unsigned long long id = blockIdx.x * blockDim.x + threadIdx.x;
            __shared__ double temp[THREADS_PER_BLOCK];
            //due to the shared directive every block gets its own copy of tmp
            double tmp = 0.0;
            while (id < size)
            {
                tmp += mat[id]*mat[id];
                id += blockDim.x * gridDim.x;
            }
            temp[tid] = tmp;
            __syncthreads();

            // for reductions, threadsPerBlock must be a power of 2
            // because of the following code
            int i = blockDim.x / 2;
            while (i != 0)
            {
                if (tid < i)

                    temp[tid] += temp[tid + i];

                __syncthreads();

                i /= 2;
            }
            // Thread 0 adds partial sums to overall sum
            if (tid == 0)
                sum[blockIdx.x] = temp[0]; 
        }

        template<typename T>
        void FrobeniusKernel(const CudaRuntime& cudart, unsigned long long ndim, const T* mat, T* vec)
        {
            int gridS = cudart.gridSize(ndim, 1);
            Frobenius<<<gridS, cudart.blockSize(), cudart.blockSize()*sizeof(float), cudart.getStream()>>>(ndim, mat, vec);
            
        }

        template void FrobeniusKernel(const CudaRuntime& cudart, unsigned long long ndim, const float* mat, float* vec); 
        template void FrobeniusKernel(const CudaRuntime& cudart, unsigned long long ndim, const double* mat, double* vec);  
    } // namespace gpu
    
} // namespace tcgmtensor
