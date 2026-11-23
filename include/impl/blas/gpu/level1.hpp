#pragma once
#include "const.h"
#include "linalg.hpp"
#include "runtime.hpp"


namespace lahva{
    namespace gpu{
    double InnerVectorProduct(const CudaRuntime& cudart, const Tensor_<double>& X, const Tensor_<double>& Y);
    float InnerVectorProduct(const CudaRuntime& cudart, const Tensor_<float>& X, const Tensor_<float>& Y);
    double InnerVectorProduct(const CudaRuntime& cudart, const Tensor_<double>& X, const size_t strideX, const Tensor_<double>& Y, const size_t strideY);
    float InnerVectorProduct(const CudaRuntime& cudart, const Tensor_<float>& X, const size_t strideX, const Tensor_<float>& Y, const size_t strideY);

    void AddVectors(const CudaRuntime& cudart, double a, const Tensor_<double>& x, Tensor_<double>& y);
    void AddVectors(const CudaRuntime& cudart, float a, const Tensor_<float>& x, Tensor_<float>& y);
    void AddVectors(const CudaRuntime& cudart, double a, const Tensor_<double>& x, size_t ix, Tensor_<double>& y, size_t iy);
    void AddVectors(const CudaRuntime& cudart, float a, const Tensor_<float>& x, size_t ix, Tensor_<float>& y, size_t iy);
    void AddVectors(const CudaRuntime& cudart, double a, const Tensor_<double>& x, Tensor_<float>& y);
    void AddVectors(const CudaRuntime& cudart, double a, const Tensor_<float>& x, Tensor_<double>& y);
    void AddVectors(const CudaRuntime& cudart, float a, const Tensor_<__half>& x, Tensor_<float>& y);
    void AddVectors(const CudaRuntime& cudart, double a, const Tensor_<__half>& x, Tensor_<double>& y);

    void CopyVectors(const CudaRuntime& cudart, const Tensor_<double>& X, Tensor_<double>& Y);
    void CopyVectors(const CudaRuntime& cudart, const Tensor_<float>& X, Tensor_<float>& Y);
    void CopyVectors(const CudaRuntime& cudart, const Tensor_<double>& X, Tensor_<float>& Y);
    void CopyVectors(const CudaRuntime& cudart, const Tensor_<float>& X, Tensor_<double>& Y);
    void CopyVectors(const CudaRuntime& cudart, const Tensor_<double>& X, const size_t strideX, Tensor_<double>& Y, const size_t strideY);
    void CopyVectors(const CudaRuntime& cudart, const Tensor_<float>& X, const size_t strideX, Tensor_<float>& Y, const size_t strideY);

    void SwapVectors(const CudaRuntime& cudart, Tensor_<double>& X, Tensor_<double>& Y);
    void SwapVectors(const CudaRuntime& cudart, Tensor_<float>& X, Tensor_<float>& Y);
    void SwapVectors(const CudaRuntime& cudart, Tensor_<double>& X, const size_t strideX, Tensor_<double>& Y, const size_t strideY);
    void SwapVectors(const CudaRuntime& cudart, Tensor_<float>& X, const size_t strideX, Tensor_<float>& Y, const size_t strideY);

    void ScaleVector(const CudaRuntime& cudart, double a, Tensor_<double>& x);
    void ScaleVector(const CudaRuntime& cudart, float a, Tensor_<float>& x);
    void ScaleVector(const CudaRuntime& cudart, double a, Tensor_<double>& x, size_t ix);
    void ScaleVector(const CudaRuntime& cudart, float a, Tensor_<float>& x, size_t ix);

    template<typename inprec, typename outprec>
    void AddVectors(const CudaRuntime& cudart, const Tensor_<inprec>& in, Tensor_<outprec>& out);

    } // namespace gpu

} 
