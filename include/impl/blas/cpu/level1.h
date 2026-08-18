#ifndef LAHVA_C_LEVEL1_H
#define LAHVA_C_LEVEL1_H
#include "const.h"

namespace lahva{
    namespace cpu{ 
    double InnerVectorProduct(const size_t nelemXY, const double* X, const double* Y);
    float InnerVectorProduct(const size_t nelemXY, const float* X, const float* Y);
    double InnerVectorProduct(const size_t nelemXY, const double* X, const size_t strideX, const double* Y, const size_t strideY);
    float InnerVectorProduct(const size_t nelemXY, const float* X, const size_t strideX, const float* Y, const size_t strideY);

    void AddVectors(const size_t ndim, const double a, const double* x, double* y);
    void AddVectors(const size_t ndim, const float a, const float* x, float* y);
    void AddVectors(const size_t ndim, const double a, const double* x, size_t ix, double* y, size_t iy);
    void AddVectors(const size_t ndim, const float a, const float* x, size_t ix, float* y, size_t iy);

    void CopyVectors(const size_t nelemXY, const double* X, double* Y);
    void CopyVectors(const size_t nelemXY, const float* X, float* Y);
    void CopyVectors(const size_t nelemXY, const double* X, const size_t strideX, double* Y, const size_t strideY);
    void CopyVectors(const size_t nelemXY, const float* X, const size_t strideX, float* Y, const size_t strideY);
    void CopyVectors(const size_t nelemXY, const double* X, float* Y);
    void CopyVectors(const size_t nelemXY, const float* X, double* Y);

    void SwapVectors(const size_t nelemXY, double* X, double* Y);
    void SwapVectors(const size_t nelemXY, float* X, float* Y);
    void SwapVectors(const size_t nelemXY, double* X, const size_t strideX, double* Y, const size_t strideY);
    void SwapVectors(const size_t nelemXY, float* X, const size_t strideX, float* Y, const size_t strideY);

    void ScaleVector(const size_t ndim, const double a, double* x);
    void ScaleVector(const size_t ndim, const float a, float* x);
    void ScaleVector(const size_t ndim, const double a, double* x, size_t ix);
    void ScaleVector(const size_t ndim, const float a, float* x, size_t ix);

    int IndexMaxFromVector(const size_t ndim, const double* x);
    int IndexMaxFromVector(const size_t ndim, const float* x);
    int IndexMaxFromVector(const size_t ndim, const double* x, const size_t ix);
    int IndexMaxFromVector(const size_t ndim, const float* x, const size_t ix);

    }
}

#endif