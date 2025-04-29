#ifndef LAHVA_GPU_LEVEL1_HPP
#define LAHVA_GPU_LEVEL1_HPP
#include "const.h"
#include "linalg.hpp"
#include "runtime.hpp"


namespace lahva{
    namespace gpu{
    double InnerVectorProduct(const CudaRuntime& cudart, const GPUTensor_<double>& X, const GPUTensor_<double>& Y);
    float InnerVectorProduct(const CudaRuntime& cudart, const GPUTensor_<float>& X, const GPUTensor_<float>& Y);
    double InnerVectorProduct(const CudaRuntime& cudart, const GPUTensor_<double>& X, const size_t strideX, const GPUTensor_<double>& Y, const size_t strideY);
    float InnerVectorProduct(const CudaRuntime& cudart, const GPUTensor_<float>& X, const size_t strideX, const GPUTensor_<float>& Y, const size_t strideY);

    void AddVectors(const CudaRuntime& cudart, double a, const GPUTensor_<double>& x, GPUTensor_<double>& y);
    void AddVectors(const CudaRuntime& cudart, float a, const GPUTensor_<float>& x, GPUTensor_<float>& y);
    void AddVectors(const CudaRuntime& cudart, double a, const GPUTensor_<double>& x, size_t ix, GPUTensor_<double>& y, size_t iy);
    void AddVectors(const CudaRuntime& cudart, float a, const GPUTensor_<float>& x, size_t ix, GPUTensor_<float>& y, size_t iy);
    void AddVectors(const CudaRuntime& cudart, double a, const GPUTensor_<double>& x, GPUTensor_<float>& y);
    void AddVectors(const CudaRuntime& cudart, double a, const GPUTensor_<float>& x, GPUTensor_<double>& y);

    void CopyVectors(const CudaRuntime& cudart, const GPUTensor_<double>& X, GPUTensor_<double>& Y);
    void CopyVectors(const CudaRuntime& cudart, const GPUTensor_<float>& X, GPUTensor_<float>& Y);
    void CopyVectors(const CudaRuntime& cudart, const GPUTensor_<double>& X, GPUTensor_<float>& Y);
    void CopyVectors(const CudaRuntime& cudart, const GPUTensor_<float>& X, GPUTensor_<double>& Y);
    void CopyVectors(const CudaRuntime& cudart, const GPUTensor_<double>& X, const size_t strideX, GPUTensor_<double>& Y, const size_t strideY);
    void CopyVectors(const CudaRuntime& cudart, const GPUTensor_<float>& X, const size_t strideX, GPUTensor_<float>& Y, const size_t strideY);

    void SwapVectors(const CudaRuntime& cudart, GPUTensor_<double>& X, GPUTensor_<double>& Y);
    void SwapVectors(const CudaRuntime& cudart, GPUTensor_<float>& X, GPUTensor_<float>& Y);
    void SwapVectors(const CudaRuntime& cudart, GPUTensor_<double>& X, const size_t strideX, GPUTensor_<double>& Y, const size_t strideY);
    void SwapVectors(const CudaRuntime& cudart, GPUTensor_<float>& X, const size_t strideX, GPUTensor_<float>& Y, const size_t strideY);

    void ScaleVector(const CudaRuntime& cudart, double a, GPUTensor_<double>& x);
    void ScaleVector(const CudaRuntime& cudart, float a, GPUTensor_<float>& x);
    void ScaleVector(const CudaRuntime& cudart, double a, GPUTensor_<double>& x, size_t ix);
    void ScaleVector(const CudaRuntime& cudart, float a, GPUTensor_<float>& x, size_t ix);

    template<typename inprec, typename outprec>
    void AddVectors(const CudaRuntime& cudart, const GPUTensor_<inprec>& in, GPUTensor_<outprec>& out);

    }
} 

#endif