#include "linalg.hpp"
#include "runtime.hpp"
#include "impl/blas/gpu/additional-level1.hpp"
#include "../gpu-utils/utils.hpp"
#include "reductions/reduction.cuh"
#include "reductions/common.cuh"
#include "additional-level1.cuh"
#include "./additional-level1.hpp"
#include "add-vectors.hpp"
#include "custom-kernel/common.h"
namespace lahva
{
    namespace gpu
    {
        __global__ void AddVector2D_(unsigned long long nrows_a, unsigned long long ncols_a, unsigned long long nrows_b, unsigned long long ncols_b, const double alpha, const float *a, double *b)
        {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            if (row < nrows_b && col < ncols_b)
            {
                b[row * ncols_b + col] = getFMA<double>(a[row * ncols_a + col], alpha, b[row * ncols_b + col]);
            }
        };

        __global__ void AddVector2D_(unsigned long long nrows_a, unsigned long long ncols_a, unsigned long long nrows_b, unsigned long long ncols_b, const float alpha, const float *a, float *b)
        {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            if (row < nrows_b && col < ncols_b)
            {
                b[row * ncols_b + col] = getFMA<float>(a[row * ncols_a + col], alpha, b[row * ncols_b + col]);
            }
        };

         __global__ void AddVector2D_(unsigned long long nrows_a, unsigned long long ncols_a, unsigned long long nrows_b, unsigned long long ncols_b, const double * alpha, const float *a, double *b)
        {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            if (row < nrows_b && col < ncols_b)
            {
                b[row * ncols_b + col] = getFMA<double>(a[row * ncols_a + col], *alpha, b[row * ncols_b + col]);
            }
        };

         __global__ void AddVector1D_(unsigned long long ntot, const double * alpha, const float *a, double *b)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            const size_t stride = blockDim.x * gridDim.x;
            for (size_t i = idx; i < ntot/2; i += stride)
            {
                float2 val_a = reinterpret_cast<const float2*>(a)[i];
                double2 val_a_d =  reinterpret_cast<double2*>(b)[i];
                val_a_d.x += static_cast<double>(val_a.x)* (*alpha);
                val_a_d.y += static_cast<double>(val_a.y)* (*alpha);
                reinterpret_cast<double2*>(b)[i] = val_a_d;
            }
        };

        __global__ void AddVector2D_(unsigned long long nrows_a, unsigned long long ncols_a, unsigned long long nrows_b, unsigned long long ncols_b, const float * alpha, const float *a, float *b)
        {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            if (row < nrows_b && col < ncols_b)
            {
                b[row * ncols_b + col] = getFMA<float>(a[row * ncols_a + col], *alpha, b[row * ncols_b + col]);
            }
        };

        void AddMatrix(const CudaRuntime& cudart, const double alpha, const Matrix_<float>& a, Matrix_<double>& b)
        {
            check_device_alloc(cudart, a);
            check_device_alloc(cudart, b);
            cudart.setDevice();
            dim3 blockSize(16,16);
            dim3 gridSize(ceil((double)(a.shape().second) / double(blockSize.x)), ceil((double)(a.shape().first) / double(blockSize.y)));
            AddVector2D_<<<gridSize, blockSize, 0, cudart.getStream()>>>
            (a.shape().first, a.shape().second, b.shape().first, b.shape().second, alpha, a.gpu_data(), b.gpu_data());
            
        }

        void AddMatrix(const CudaRuntime& cudart, const float alpha, const Matrix_<float>& a, Matrix_<float>& b)
        {
            check_device_alloc(cudart, a);
            check_device_alloc(cudart, b);
            cudart.setDevice();
            dim3 blockSize(16,16);
            dim3 gridSize(ceil((double)(a.shape().second) / double(blockSize.x)), ceil((double)(a.shape().first) / double(blockSize.y)));
            AddVector2D_<<<gridSize, blockSize, 0, cudart.getStream()>>>
            (a.shape().first, a.shape().second, b.shape().first, b.shape().second, alpha, a.gpu_data(), b.gpu_data());
            
        }

        void AddMatrix(const CudaRuntime& cudart, const double* alpha, const Matrix_<float>& a, Matrix_<double>& b)
        {
            check_device_alloc(cudart, a);
            check_device_alloc(cudart, b);
            cudart.setDevice();
            if (a.shape() == b.shape())
            {
                
                size_t blockSize = cudart.blockSize();
                size_t numBlocks = cudart.gridSize(a.size(), 1);
                AddVector1D_<<<numBlocks/4, blockSize, 0, cudart.getStream()>>>
                (a.size(), alpha, a.gpu_data(), b.gpu_data());
                return;
            }
            dim3 blockSize(16,32);
            dim3 gridSize(ceil((double)(a.shape().second) / double(blockSize.x)), ceil((double)(a.shape().first) / double(blockSize.y)));
            AddVector2D_<<<gridSize, blockSize, 0, cudart.getStream()>>>
            (a.shape().first, a.shape().second, b.shape().first, b.shape().second, alpha, a.gpu_data(), b.gpu_data());
            
        }

        void AddMatrix(const CudaRuntime& cudart, const float* alpha, const Matrix_<float>& a, Matrix_<float>& b)
        {
            check_device_alloc(cudart, a);
            check_device_alloc(cudart, b);
            cudart.setDevice();
            dim3 blockSize(16,32);
            dim3 gridSize(ceil((double)(a.shape().second) / double(blockSize.x)), ceil((double)(a.shape().first) / double(blockSize.y)));
            AddVector2D_<<<gridSize, blockSize, 0, cudart.getStream()>>>
            (a.shape().first, a.shape().second, b.shape().first, b.shape().second, alpha, a.gpu_data(), b.gpu_data());
            
        }
    } // namespace gpu
    
} // namespace lahva
