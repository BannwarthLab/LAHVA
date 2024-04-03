#ifndef TCGMBLAS_C_LEVEL1_HPP
#define TCGMBLAS_C_LEVEL1_HPP
#include "const.h"
#include "linalg.hpp"

namespace tcgmtensor{
    
    double InnerVectorProduct(const Vector<double>& X, const Vector<double>& Y);
    float InnerVectorProduct(const Vector<float>& X, const Vector<float>& Y);
    double InnerVectorProduct(const Vector<double>& X, const size_t strideX, const Vector<double>& Y, const size_t strideY);
    float InnerVectorProduct(const Vector<float>& X, const size_t strideX, const Vector<float>& Y, const size_t strideY);

    void AddVectors(const double a, const Vector<double>& x, Vector<double>& y);
    void AddVectors(const float a, const Vector<float>& x, Vector<float>& y);
    void AddVectors(const double a, const Vector<double>& x, size_t ix, Vector<double>& y, size_t iy);
    void AddVectors(const float a, const Vector<float>& x, size_t ix, Vector<float>& y, size_t iy);

    void CopyVectors(const Vector<double>& X, Vector<double>& Y);
    void CopyVectors(const Vector<float>& X, Vector<float>& Y);
    void CopyVectors(const Vector<double>& X, const size_t strideX, Vector<double>& Y, const size_t strideY);
    void CopyVectors(const Vector<float>& X, const size_t strideX, Vector<float>& Y, const size_t strideY);

    void SwapVectors(Vector<double>& X, Vector<double>& Y);
    void SwapVectors(Vector<float>& X, Vector<float>& Y);
    void SwapVectors(Vector<double>& X, const size_t strideX, Vector<double>& Y, const size_t strideY);
    void SwapVectors(Vector<float>& X, const size_t strideX, Vector<float>& Y, const size_t strideY);

    void ScaleVector(const double a, Vector<double>& x);
    void ScaleVector(const float a, Vector<float>& x);
    void ScaleVector(const double a, Vector<double>& x, size_t ix);
    void ScaleVector(const float a, Vector<float>& x, size_t ix);
} 

#endif