#include "impl/blas/gpu/additional-level1.hpp"
#include "norm.cuh"
#include "reductions/reduction.cuh"
#include "reductions/common.cuh"
#include "common.h"

namespace lahva
{
    namespace gpu
    {
        template <size_t blocks, typename T>
        __global__ static void Frobenius_(const unsigned long long size, const T *mat, T *sum)
        {
            unsigned long long tid = threadIdx.x;
            unsigned long long id = blockIdx.x * blockDim.x + threadIdx.x;
            __shared__ T temp[blocks];
            // due to the shared directive every block gets its own copy of tmp
            T tmp = 0.0;
            while (id < size)
            {
                tmp = getFMA<T>(mat[id], mat[id], tmp);
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

                temp[tid] += temp[tid + i];

                __syncthreads();

                i /= 2;
            }
            // Thread 0 adds partial sums to overall sum
            if (tid == 0)
                sum[blockIdx.x] = temp[0];
        }


        template <typename T>
        void FrobeniusKernel(const CudaRuntime &cudart, unsigned long long ndim, const T *mat, T *vec)
        {
            int gridS = cudart.gridSize(ndim, 1);
            size_t blockSize = cudart.blockSize();
            // ApplyKernel<T>(ndim, mat, 2.0, pow<T>());
            // Frobenius<T>(cudart, ndim, mat, vec, block, add_rn<T>());
            switch (blockSize)
            {
            case 1024:
                Frobenius_<1024, T><<<gridS, blockSize, 0, cudart.getStream()>>>(ndim, mat, vec);
                break;
            case 512:
                Frobenius_<512, T><<<gridS, blockSize, 0, cudart.getStream()>>>(ndim, mat, vec);
                break;
            case 256:
                Frobenius_<256, T><<<gridS, blockSize, 0, cudart.getStream()>>>(ndim, mat, vec);
                break;
            case 128:
                Frobenius_<128, T><<<gridS, blockSize, 0, cudart.getStream()>>>(ndim, mat, vec);
                break;
            case 64:
                Frobenius_<64, T><<<gridS, blockSize, 0, cudart.getStream()>>>(ndim, mat, vec);
                break;
            case 32:
                Frobenius_<32, T><<<gridS, blockSize, 0, cudart.getStream()>>>(ndim, mat, vec);
                break;
            case 16:
                Frobenius_<16, T><<<gridS, blockSize, 0, cudart.getStream()>>>(ndim, mat, vec);
                break;
            case 8:
                Frobenius_<8, T><<<gridS, blockSize, 0, cudart.getStream()>>>(ndim, mat, vec);
                break;
            case 4:
                Frobenius_<4, T><<<gridS, blockSize, 0, cudart.getStream()>>>(ndim, mat, vec);
                break;
            case 2:
                Frobenius_<2, T><<<gridS, blockSize, 0, cudart.getStream()>>>(ndim, mat, vec);
                break;
            case 1:
                Frobenius_<1, T><<<gridS, blockSize, 0, cudart.getStream()>>>(ndim, mat, vec);
                break;
            default:
                break;
            }

        };

        // template __global__ static void Frobenius_<float, 1024>(const unsigned long long size, const float* mat, float* sum);
        // template __global__ static void Frobenius_<double, 1024>(const unsigned long long size, const double* mat, double* sum);
        template void FrobeniusKernel<float>(const CudaRuntime &cudart, const unsigned long long ndim, const float *mat, float *vec);
        template void FrobeniusKernel<double>(const CudaRuntime &cudart, const unsigned long long ndim, const double *mat, double *vec);
    } // namespace gpu

} // namespace lahva
