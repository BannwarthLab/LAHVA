#ifndef TCGMBLAS_C_LEVEL1_HPP
#define TCGMBLAS_C_LEVEL1_HPP
#include "const.h"
#include "linalg.hpp"

namespace tcgmtensor{
    
    double InnerVectorProduct(const vector<double>& X, const vector<double>& Y);
    float InnerVectorProduct(const vector<float>& X, const vector<float>& Y);
    double InnerVectorProduct(const vector<double>& X, const size_t strideX, const vector<double>& Y, const size_t strideY);
    float InnerVectorProduct(const vector<float>& X, const size_t strideX, const vector<float>& Y, const size_t strideY);

    void AddVectors(const double a, const vector<double>& x, vector<double>& y);
    void AddVectors(const float a, const vector<float>& x, vector<float>& y);
    void AddVectors(const double a, const vector<double>& x, size_t ix, vector<double>& y, size_t iy);
    void AddVectors(const float a, const vector<float>& x, size_t ix, vector<float>& y, size_t iy);

    void CopyVectors(const vector<double>& X, vector<double>& Y);
    void CopyVectors(const vector<float>& X, vector<float>& Y);
    void CopyVectors(const vector<double>& X, const size_t strideX, vector<double>& Y, const size_t strideY);
    void CopyVectors(const vector<float>& X, const size_t strideX, vector<float>& Y, const size_t strideY);

    void SwapVectors(vector<double>& X, vector<double>& Y);
    void SwapVectors(vector<float>& X, vector<float>& Y);
    void SwapVectors(vector<double>& X, const size_t strideX, vector<double>& Y, const size_t strideY);
    void SwapVectors(vector<float>& X, const size_t strideX, vector<float>& Y, const size_t strideY);

    void ScaleVector(const double a, vector<double>& x);
    void ScaleVector(const float a, vector<float>& x);
    void ScaleVector(const double a, vector<double>& x, size_t ix);
    void ScaleVector(const float a, vector<float>& x, size_t ix);
} 

#endif