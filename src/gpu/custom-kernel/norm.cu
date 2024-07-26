#include "impl/gpu/additional-level1.hpp"

namespace tcgmtensor
{
    namespace gpu
    {
        __global__ static void sFrobenius(const unsigned long long ndim, const float* mat, const float* sum)
        {
            unsigned long long tid = threadIdx.x;
            unsigned long long id = blockIdx.x * blockDim.x + threadIdx.x;
            __shared__ float temp[THREADS_PER_BLOCK];
            //due to the shared directive every block gets its own copy of tmp
            float tmp = 0.0;
            while (id < ndim*ndim)
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

        __global__ static void dFrobenius(const unsigned long long ndim, const double* mat, const double* sum)
        {
            unsigned long long tid = threadIdx.x;
            unsigned long long id = blockIdx.x * blockDim.x + threadIdx.x;
            __shared__ double temp[THREADS_PER_BLOCK];
            //due to the shared directive every block gets its own copy of tmp
            double tmp = 0.0;
            while (id < ndim*ndim)
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

        template<>
        float FrobeniusNorm(const CudaRuntime& cudart, const Matrix<float>& mat)
        {
            check_device_alloc(cudart, mat);
            int gridS = cudart.gridSize(mat.size(), 1);
            Vector<float> vec(gridS);
            vec.copy2device(cudart);

            sFrobenius<<<gridS, cudart.blockSize()>>>(mat.shape().first, mat.gpu_data(), vec.gpu_data());
            float norm = vec.sum();
            
            return std::sqrt(norm);
        }

        template<>
        double FrobeniusNorm(const CudaRuntime& cudart, const Matrix<double>& mat)
        {
            check_device_alloc(cudart, mat);
            int gridS = cudart.gridSize(mat.size(), 1);
            Vector<float> vec(gridS);
            vec.copy2device(cudart);

            dFrobenius<<<gridS, cudart.blockSize()>>>(mat.shape().first, mat.gpu_data(), vec.gpu_data());
            double norm = vec.sum();
            
            return std::sqrt(norm);
        }
    } // namespace gpu
    
} // namespace tcgmtensor
