#ifndef TCGMBLAS_GPU_LEVEL1_HPP
#define TCGMBLAS_GPU_LEVEL1_HPP
#include "const.h"
#include "linalg.hpp"
#include "runtime.hpp"


namespace tcgmtensor{
    namespace gpu{
    double InnerVectorProduct(const CudaRuntime& cudart, const GPUTensor<double>& X, const GPUTensor<double>& Y);
    float InnerVectorProduct(const CudaRuntime& cudart, const GPUTensor<float>& X, const GPUTensor<float>& Y);
    double InnerVectorProduct(const CudaRuntime& cudart, const GPUTensor<double>& X, const size_t strideX, const GPUTensor<double>& Y, const size_t strideY);
    float InnerVectorProduct(const CudaRuntime& cudart, const GPUTensor<float>& X, const size_t strideX, const GPUTensor<float>& Y, const size_t strideY);

    void AddVectors(const CudaRuntime& cudart, double a, const GPUTensor<double>& x, GPUTensor<double>& y);
    void AddVectors(const CudaRuntime& cudart, float a, const GPUTensor<float>& x, GPUTensor<float>& y);
    void AddVectors(const CudaRuntime& cudart, double a, const GPUTensor<double>& x, size_t ix, GPUTensor<double>& y, size_t iy);
    void AddVectors(const CudaRuntime& cudart, float a, const GPUTensor<float>& x, size_t ix, GPUTensor<float>& y, size_t iy);

    void CopyVectors(const CudaRuntime& cudart, const GPUTensor<double>& X, GPUTensor<double>& Y);
    void CopyVectors(const CudaRuntime& cudart, const GPUTensor<float>& X, GPUTensor<float>& Y);
    void CopyVectors(const CudaRuntime& cudart, const GPUTensor<double>& X, const size_t strideX, GPUTensor<double>& Y, const size_t strideY);
    void CopyVectors(const CudaRuntime& cudart, const GPUTensor<float>& X, const size_t strideX, GPUTensor<float>& Y, const size_t strideY);

    void SwapVectors(const CudaRuntime& cudart, GPUTensor<double>& X, GPUTensor<double>& Y);
    void SwapVectors(const CudaRuntime& cudart, GPUTensor<float>& X, GPUTensor<float>& Y);
    void SwapVectors(const CudaRuntime& cudart, GPUTensor<double>& X, const size_t strideX, GPUTensor<double>& Y, const size_t strideY);
    void SwapVectors(const CudaRuntime& cudart, GPUTensor<float>& X, const size_t strideX, GPUTensor<float>& Y, const size_t strideY);

    void ScaleVector(const CudaRuntime& cudart, double a, GPUTensor<double>& x);
    void ScaleVector(const CudaRuntime& cudart, float a, GPUTensor<float>& x);
    void ScaleVector(const CudaRuntime& cudart, double a, GPUTensor<double>& x, size_t ix);
    void ScaleVector(const CudaRuntime& cudart, float a, GPUTensor<float>& x, size_t ix);
    }
} 

#endif