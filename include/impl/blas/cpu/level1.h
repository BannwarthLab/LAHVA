#ifndef LAHVA_C_LEVEL1_H
#define LAHVA_C_LEVEL1_H
#include "const.h"

// In general, the functions declared here are C-style BLAS level 1 functions
namespace lahva
{
    namespace cpu
    {
        /// @brief Computes the inner product of two vectors, wrapper to BLAS function ddot
        /// @param nelemXY Number of elements in vectors X and Y
        /// @param X Pointer to the first element of vector X, stride 1 assumed
        /// @param Y Pointer to the first element of vector Y, stride 1 assumed
        /// @return The inner product of vectors X and Y
        double InnerVectorProduct(const size_t nelemXY, const double *X, const double *Y);

        /// @brief Computes the inner product of two vectors, wrapper to BLAS function sdot
        /// @param nelemXY Number of elements in vectors X and Y
        /// @param X Pointer to the first element of vector X, stride 1 assumed
        /// @param Y Pointer to the first element of vector Y, stride 1 assumed
        /// @return The inner product of vectors X and Y
        float InnerVectorProduct(const size_t nelemXY, const float *X, const float *Y);

        /// @brief Computes the inner product of two vectors with strides, wrapper to BLAS function ddot
        /// @param nelemXY Number of elements in vectors X and Y
        /// @param X Pointer to the first element of vector X
        /// @param strideX Stride between elements in vector X
        /// @param Y Pointer to the first element of vector Y
        /// @param strideY Stride between elements in vector Y
        /// @return The inner product of vectors X and Y
        double InnerVectorProduct(const size_t nelemXY, const double *X, const size_t strideX, const double *Y, const size_t strideY);

        /// @brief Computes the inner product of two vectors with strides, wrapper to BLAS function sdot
        /// @param nelemXY Number of elements in vectors X and Y
        /// @param X Pointer to the first element of vector X
        /// @param strideX Stride between elements in vector X
        /// @param Y Pointer to the first element of vector Y
        /// @param strideY Stride between elements in vector Y
        /// @return The inner product of vectors X and Y
        float InnerVectorProduct(const size_t nelemXY, const float *X, const size_t strideX, const float *Y, const size_t strideY);

        /// @brief Adds a scaled vector to another vector, wrapper to BLAS function daxpy
        /// @param ndim Number of elements in vectors x and y
        /// @param a Scalar multiplier for vector x
        /// @param x Pointer to the first element of vector x, stride 1 assumed
        /// @param y Pointer to the first element of vector y, stride 1 assumed
        void AddVectors(const size_t ndim, const double a, const double *x, double *y);

        /// @brief Adds a scaled vector to another vector, wrapper to BLAS function saxpy
        /// @param ndim Number of elements in vectors x and y
        /// @param a Scalar multiplier for vector x
        /// @param x Pointer to the first element of vector x, stride 1 assumed
        /// @param y Pointer to the first element of vector y, stride 1 assumed
        void AddVectors(const size_t ndim, const float a, const float *x, float *y);

        /// @brief Adds a scaled vector to another vector with strides, wrapper to BLAS function daxpy
        /// @param ndim Number of elements in vectors x and y
        /// @param a Scalar multiplier for vector x
        /// @param x Pointer to the first element of vector x
        /// @param ix Stride between elements in vector x
        /// @param y Pointer to the first element of vector y
        /// @param iy Stride between elements in vector y
        void AddVectors(const size_t ndim, const double a, const double *x, size_t ix, double *y, size_t iy);

        /// @brief Adds a scaled vector to another vector with strides, wrapper to BLAS function saxpy
        /// @param ndim Number of elements in vectors x and y
        /// @param a Scalar multiplier for vector x
        /// @param x Pointer to the first element of vector x
        /// @param ix Stride between elements in vector x
        /// @param y Pointer to the first element of vector y
        /// @param iy Stride between elements in vector y
        void AddVectors(const size_t ndim, const float a, const float *x, size_t ix, float *y, size_t iy);

        /// @brief Copies elements from one vector to another, wrapper to BLAS function dcopy
        /// @param nelemXY Number of elements in vectors X and Y
        /// @param X Pointer to the first element of vector X, stride 1 assumed
        /// @param Y Pointer to the first element of vector Y, stride 1 assumed
        void CopyVectors(const size_t nelemXY, const double *X, double *Y);

        /// @brief Copies elements from one vector to another, wrapper to BLAS function scopy
        /// @param nelemXY Number of elements in vectors X and Y
        /// @param X Pointer to the first element of vector X, stride 1 assumed
        /// @param Y Pointer to the first element of vector Y, stride 1 assumed
        void CopyVectors(const size_t nelemXY, const float *X, float *Y);

        /// @brief Copies elements from one vector to another with strides, wrapper to BLAS function dcopy
        /// @param nelemXY Number of elements in vectors X and Y
        /// @param X Pointer to the first element of vector X
        /// @param strideX Stride between elements in vector X
        /// @param Y Pointer to the first element of vector Y
        /// @param strideY Stride between elements in vector Y
        void CopyVectors(const size_t nelemXY, const double *X, const size_t strideX, double *Y, const size_t strideY);

        /// @brief Copies elements from one vector to another with strides, wrapper to BLAS function scopy
        /// @param nelemXY Number of elements in vectors X and Y
        /// @param X Pointer to the first element of vector X
        /// @param strideX Stride between elements in vector X
        /// @param Y Pointer to the first element of vector Y
        /// @param strideY Stride between elements in vector Y
        void CopyVectors(const size_t nelemXY, const float *X, const size_t strideX, float *Y, const size_t strideY);

        /// @brief Copies elements from one vector to another, Mixed precision version FP64(X) -> FP32(Y)
        /// @param nelemXY Number of elements in vectors X and Y
        /// @param X Pointer to the first element of vector X, stride 1 assumed
        /// @param Y Pointer to the first element of vector Y, stride 1 assumed
        void CopyVectors(const size_t nelemXY, const double *X, float *Y);

        /// @brief Copies elements from one vector to another, Mixed precision version FP32(X) -> FP64(Y)
        /// @param nelemXY Number of elements in vectors X and Y
        /// @param X Pointer to the first element of vector X, stride 1 assumed
        /// @param Y Pointer to the first element of vector Y, stride 1 assumed
        void CopyVectors(const size_t nelemXY, const float *X, double *Y);

        /// @brief Swaps elements between two vectors, wrapper to BLAS function dswap
        /// @param nelemXY Number of elements in vectors X and Y
        /// @param X Pointer to the first element of vector X, stride 1 assumed
        /// @param Y Pointer to the first element of vector Y, stride 1 assumed
        void SwapVectors(const size_t nelemXY, double *X, double *Y);

        /// @brief Swaps elements between two vectors, wrapper to BLAS function sswap
        /// @param nelemXY Number of elements in vectors X and Y
        /// @param X Pointer to the first element of vector X, stride 1 assumed
        /// @param Y Pointer to the first element of vector Y, stride 1 assumed
        void SwapVectors(const size_t nelemXY, float *X, float *Y);

        /// @brief Swaps elements between two vectors with strides, wrapper to BLAS function dswap
        /// @param nelemXY Number of elements in vectors X and Y
        /// @param X Pointer to the first element of vector X
        /// @param strideX Stride between elements in vector X
        /// @param Y Pointer to the first element of vector Y
        /// @param strideY Stride between elements in vector Y
        void SwapVectors(const size_t nelemXY, double *X, const size_t strideX, double *Y, const size_t strideY);

        /// @brief Swaps elements between two vectors with strides, wrapper to BLAS function sswap
        /// @param nelemXY Number of elements in vectors X and Y
        /// @param X Pointer to the first element of vector X
        /// @param strideX Stride between elements in vector X
        /// @param Y Pointer to the first element of vector Y
        /// @param strideY Stride between elements in vector Y
        void SwapVectors(const size_t nelemXY, float *X, const size_t strideX, float *Y, const size_t strideY);

        /// @brief Scales a vector by a constant, wrapper to BLAS function dscal
        /// @param ndim Number of elements in vector x
        /// @param a Scaling factor
        /// @param x Pointer to the first element of vector x, stride 1 assumed
        void ScaleVector(const size_t ndim, const double a, double *x);

        /// @brief Scales a vector by a constant, wrapper to BLAS function sscal
        /// @param ndim Number of elements in vector x
        /// @param a Scaling factor
        /// @param x Pointer to the first element of vector x, stride 1 assumed
        void ScaleVector(const size_t ndim, const float a, float *x);

        /// @brief Scales a vector by a constant with stride, wrapper to BLAS function dscal
        /// @param ndim Number of elements in vector x
        /// @param a Scaling factor
        /// @param x Pointer to the first element of vector x
        /// @param ix Stride between elements in vector x
        void ScaleVector(const size_t ndim, const double a, double *x, size_t ix);

        /// @brief Scales a vector by a constant with stride, wrapper to BLAS function sscal
        /// @param ndim Number of elements in vector x
        /// @param a Scaling factor
        /// @param x Pointer to the first element of vector x
        /// @param ix Stride between elements in vector x
        void ScaleVector(const size_t ndim, const float a, float *x, size_t ix);

        /// @brief Finds the index of the maximum element in a vector, wrapper to BLAS function idamax
        /// @param ndim Number of elements in vector x
        /// @param x Pointer to the first element of vector x, stride 1 assumed
        /// @return Index of the maximum element in vector x
        int IndexMaxFromVector(const size_t ndim, const double *x);

        /// @brief Finds the index of the maximum element in a vector, wrapper to BLAS function isamax
        /// @param ndim Number of elements in vector x
        /// @param x Pointer to the first element of vector x, stride 1 assumed
        /// @return Index of the maximum element in vector x
        int IndexMaxFromVector(const size_t ndim, const float *x);

        /// @brief Finds the index of the maximum element in a vector with stride, wrapper to BLAS function idamax
        /// @param ndim Number of elements in vector x
        /// @param x Pointer to the first element of vector x
        /// @param ix Stride between elements in vector x
        /// @return Index of the maximum element in vector x
        int IndexMaxFromVector(const size_t ndim, const double *x, const size_t ix);

        /// @brief Finds the index of the maximum element in a vector with stride, wrapper to BLAS function isamax
        /// @param ndim Number of elements in vector x
        /// @param x Pointer to the first element of vector x
        /// @param ix Stride between elements in vector x
        /// @return Index of the maximum element in vector x
        int IndexMaxFromVector(const size_t ndim, const float *x, const size_t ix);

    }
}

#endif