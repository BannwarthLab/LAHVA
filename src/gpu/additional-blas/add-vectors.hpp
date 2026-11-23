#pragma once
#include "linalg.hpp"

namespace lahva
{
    namespace gpu
    {
        void AddVector(const CudaRuntime& cudart, unsigned long long ndim, const double a, const Tensor_<double>& x, Tensor_<float>& y);
        void AddVector(const CudaRuntime& cudart, unsigned long long ndim, const double a, const Tensor_<float>& x, Tensor_<double>& y);
        void AddVector(const CudaRuntime& cudart, unsigned long long ndim, const double a, const Tensor_<__half>& x, Tensor_<double>& y);
    } // namespace gpu

} // namespace lahva
