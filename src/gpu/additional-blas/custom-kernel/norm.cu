#include "impl/blas/gpu/additional-level1.hpp"
#include "norm.cuh"
#include "reductions/reduction.cuh"
#include "reductions/common.cuh"
#include "common.h"
#include "linalg.hpp"

namespace lahva
{
    namespace gpu
    {
       

        
        template <typename T>
        void FrobeniusKernel(const CudaRuntime &cudart, unsigned long long ndim, const T *mat, T *vec)
        {
            int gridS = cudart.gridSize(ndim, 1);
            size_t blockSize = cudart.blockSize();
 
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

        template <typename T>
        void FrobeniusKernel2(const CudaRuntime &cudart, unsigned long long ndim, const T *mat1, const T* mat2, T *vec)
        {
            int gridS = cudart.gridSize(ndim, 1);
            size_t blockSize = cudart.blockSize();
 
            switch (blockSize)
            {
            case 1024:
                Frobenius_2<1024, T><<<gridS, blockSize, 0, cudart.getStream()>>>(ndim, mat1, mat2, vec);
                break;
            case 512:
                Frobenius_2<512, T><<<gridS, blockSize, 0, cudart.getStream()>>>(ndim, mat1, mat2, vec);
                break;
            case 256:
                Frobenius_2<256, T><<<gridS, blockSize, 0, cudart.getStream()>>>(ndim, mat1, mat2, vec);
                break;
            case 128:
                Frobenius_2<128, T><<<gridS, blockSize, 0, cudart.getStream()>>>(ndim, mat1, mat2, vec);
                break;
            case 64:
                Frobenius_2<64, T><<<gridS, blockSize, 0, cudart.getStream()>>>(ndim, mat1, mat2, vec);
                break;
            case 32:
                Frobenius_2<32, T><<<gridS, blockSize, 0, cudart.getStream()>>>(ndim, mat1, mat2, vec);
                break;
            case 16:
                Frobenius_2<16, T><<<gridS, blockSize, 0, cudart.getStream()>>>(ndim, mat1, mat2, vec);
                break;
            case 8:
                Frobenius_2<8, T><<<gridS, blockSize, 0, cudart.getStream()>>>(ndim, mat1, mat2, vec);
                break;
            case 4:
                Frobenius_2<4, T><<<gridS, blockSize, 0, cudart.getStream()>>>(ndim, mat1, mat2, vec);
                break;
            case 2:
                Frobenius_2<2, T><<<gridS, blockSize, 0, cudart.getStream()>>>(ndim, mat1, mat2, vec);
                break;
            case 1:
                Frobenius_2<1, T><<<gridS, blockSize, 0, cudart.getStream()>>>(ndim, mat1, mat2, vec);
                break;
            default:
                break;
            }

        };

        // template __global__ static void Frobenius_<float, 1024>(const unsigned long long size, const float* mat, float* sum);
        // template __global__ static void Frobenius_<double, 1024>(const unsigned long long size, const double* mat, double* sum);
        template void FrobeniusKernel<float>(const CudaRuntime &cudart, const unsigned long long ndim, const float *mat, float *vec);
        template void FrobeniusKernel<double>(const CudaRuntime &cudart, const unsigned long long ndim, const double *mat, double *vec);
        template void FrobeniusKernel2<float>(const CudaRuntime &cudart, const unsigned long long ndim, const float *mat1, const float* mat2, float *vec);
        template void FrobeniusKernel2<double>(const CudaRuntime &cudart, const unsigned long long ndim, const double *mat1, const double * mat2, double *vec);
    } // namespace gpu

} // namespace lahva
