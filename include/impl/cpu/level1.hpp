#ifndef TCGMBLAS_C_LEVEL1_HPP
#define TCGMBLAS_C_LEVEL1_HPP
#include "const.h"
#include "linalg.hpp"

namespace tcgmtensor{
    namespace cpu{
    double InnerVectorProduct(const Tensor<double>& X, const Tensor<double>& Y);
    float InnerVectorProduct(const Tensor<float>& X, const Tensor<float>& Y);
    double InnerVectorProduct(const Tensor<double>& X, const size_t strideX, const Tensor<double>& Y, const size_t strideY);
    float InnerVectorProduct(const Tensor<float>& X, const size_t strideX, const Tensor<float>& Y, const size_t strideY);

    void AddVectors(const double a, const Tensor<double>& x, Tensor<double>& y);
    void AddVectors(const float a, const Tensor<float>& x, Tensor<float>& y);
    void AddVectors(const double a, const Tensor<double>& x, size_t ix, Tensor<double>& y, size_t iy);
    void AddVectors(const float a, const Tensor<float>& x, size_t ix, Tensor<float>& y, size_t iy);

    void CopyVectors(const Tensor<double>& X, Tensor<double>& Y);
    void CopyVectors(const Tensor<float>& X, Tensor<float>& Y);
    void CopyVectors(const Tensor<double>& X, const size_t strideX, Tensor<double>& Y, const size_t strideY);
    void CopyVectors(const Tensor<float>& X, const size_t strideX, Tensor<float>& Y, const size_t strideY);
    void CopyVectors(const Tensor<float>& X, Tensor<double>& Y);
    void CopyVectors(const Tensor<double>& X, Tensor<float>& Y);

    void SwapVectors(Tensor<double>& X, Tensor<double>& Y);
    void SwapVectors(Tensor<float>& X, Tensor<float>& Y);
    void SwapVectors(Tensor<double>& X, const size_t strideX, Tensor<double>& Y, const size_t strideY);
    void SwapVectors(Tensor<float>& X, const size_t strideX, Tensor<float>& Y, const size_t strideY);

    void ScaleVector(const double a, Tensor<double>& x);
    void ScaleVector(const float a, Tensor<float>& x);
    void ScaleVector(const double a, Tensor<double>& x, size_t ix);
    void ScaleVector(const float a, Tensor<float>& x, size_t ix);
    
    template<typename... Args>
    void InnerVectorProduct(const CPURuntime& rt_, Args&&... args) {
        (InnerVectorProduct(args...));                    
    }

    template<typename... Args>
    void AddVectors(const CPURuntime& rt_, Args&&... args) {
        (AddVectors(args...));                
    }

    template<typename... Args>
    void CopyVectors(const CPURuntime& rt_, Args&&... args) {
        (CopyVectors(args...));                
    }

    template<typename... Args>
    void SwapVectors(const CPURuntime& rt_, Args&&... args) {
        (SwapVectors(args...));                
    }
    
    template<typename... Args>
    void ScaleVector(const CPURuntime& rt_, Args&&... args) {
        (ScaleVector(args...));                
    }
    }
} 

#endif