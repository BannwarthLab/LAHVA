#ifndef LAHVA_C_LEVEL1_H
#define LAHVA_C_LEVEL1_H
#include "const.h"

// Level-1 BLAS-like operations (vector * vector) - C-style declarations
// These functions are thin wrappers around optimized CPU BLAS kernels. They operate on
// raw pointers and use simple scalar parameters for flexibility. Each overload is
// provided for double and float precision; both precisions are documented explicitly.

namespace lahva
{
    namespace cpu
    {
        /// @brief Computes the inner product of two vectors, wrapper to BLAS function ddot.
        ///
        /// Performs result = X^T * Y where X and Y are vectors of size nelemXY.
        ///
        /// @param nelemXY Number of elements in vectors X and Y.
        /// @param X Pointer to the first element of vector X (size nelemXY), stride 1 assumed.
        /// @param Y Pointer to the first element of vector Y (size nelemXY), stride 1 assumed.
        /// @return The inner product of vectors X and Y.
        double InnerVectorProduct(const size_t nelemXY, const double *X, const double *Y);

        /// @brief Computes the inner product of two vectors, wrapper to BLAS function sdot.
        ///
        /// Performs result = X^T * Y where X and Y are vectors of size nelemXY.
        ///
        /// @param nelemXY Number of elements in vectors X and Y.
        /// @param X Pointer to the first element of vector X (size nelemXY), stride 1 assumed.
        /// @param Y Pointer to the first element of vector Y (size nelemXY), stride 1 assumed.
        /// @return The inner product of vectors X and Y.
        float InnerVectorProduct(const size_t nelemXY, const float *X, const float *Y);

        /// @brief Computes the inner product of two vectors with strides, wrapper to BLAS function ddot.
        ///
        /// Performs result = X^T * Y where X and Y are vectors of size nelemXY with specified strides.
        ///
        /// @param nelemXY Number of elements in vectors X and Y.
        /// @param X Pointer to the first element of vector X (size nelemXY).
        /// @param strideX Stride between consecutive elements in vector X.
        /// @param Y Pointer to the first element of vector Y (size nelemXY).
        /// @param strideY Stride between consecutive elements in vector Y.
        /// @return The inner product of vectors X and Y.
        double InnerVectorProduct(const size_t nelemXY, const double *X, const size_t strideX, const double *Y, const size_t strideY);

        /// @brief Computes the inner product of two vectors with strides, wrapper to BLAS function sdot.
        ///
        /// Performs result = X^T * Y where X and Y are vectors of size nelemXY with specified strides.
        ///
        /// @param nelemXY Number of elements in vectors X and Y.
        /// @param X Pointer to the first element of vector X (size nelemXY).
        /// @param strideX Stride between consecutive elements in vector X.
        /// @param Y Pointer to the first element of vector Y (size nelemXY).
        /// @param strideY Stride between consecutive elements in vector Y.
        /// @return The inner product of vectors X and Y.
        float InnerVectorProduct(const size_t nelemXY, const float *X, const size_t strideX, const float *Y, const size_t strideY);

        /// @brief Adds a scaled vector to another vector, wrapper to BLAS function daxpy.
        ///
        /// Performs y = y + a * x where x and y are vectors of size ndim.
        ///
        /// @param ndim Number of elements in vectors x and y.
        /// @param a Scalar multiplier for vector x.
        /// @param x Pointer to the first element of vector x (size ndim), stride 1 assumed.
        /// @param y Pointer to the first element of vector y (size ndim), stride 1 assumed.
        void AddVectors(const size_t ndim, const double a, const double *x, double *y);

        /// @brief Adds a scaled vector to another vector, wrapper to BLAS function saxpy.
        ///
        /// Performs y = y + a * x where x and y are vectors of size ndim.
        ///
        /// @param ndim Number of elements in vectors x and y.
        /// @param a Scalar multiplier for vector x.
        /// @param x Pointer to the first element of vector x (size ndim), stride 1 assumed.
        /// @param y Pointer to the first element of vector y (size ndim), stride 1 assumed.
        void AddVectors(const size_t ndim, const float a, const float *x, float *y);

        /// @brief Adds a scaled vector to another vector with strides, wrapper to BLAS function daxpy.
        ///
        /// Performs y = y + a * x where x and y are vectors of size ndim with specified strides.
        ///
        /// @param ndim Number of elements in vectors x and y.
        /// @param a Scalar multiplier for vector x.
        /// @param x Pointer to the first element of vector x (size ndim).
        /// @param ix Stride between consecutive elements in vector x.
        /// @param y Pointer to the first element of vector y (size ndim).
        /// @param iy Stride between consecutive elements in vector y.
        void AddVectors(const size_t ndim, const double a, const double *x, size_t ix, double *y, size_t iy);

        /// @brief Adds a scaled vector to another vector with strides, wrapper to BLAS function saxpy.
        ///
        /// Performs y = y + a * x where x and y are vectors of size ndim with specified strides.
        ///
        /// @param ndim Number of elements in vectors x and y.
        /// @param a Scalar multiplier for vector x.
        /// @param x Pointer to the first element of vector x (size ndim).
        /// @param ix Stride between consecutive elements in vector x.
        /// @param y Pointer to the first element of vector y (size ndim).
        /// @param iy Stride between consecutive elements in vector y.
        void AddVectors(const size_t ndim, const float a, const float *x, size_t ix, float *y, size_t iy);

        /// @brief Copies elements from one vector to another, wrapper to BLAS function dcopy.
        ///
        /// Performs Y = X where X and Y are vectors of size nelemXY.
        ///
        /// @param nelemXY Number of elements in vectors X and Y.
        /// @param X Pointer to the first element of source vector X (size nelemXY), stride 1 assumed.
        /// @param Y Pointer to the first element of destination vector Y (size nelemXY), stride 1 assumed.
        void CopyVectors(const size_t nelemXY, const double *X, double *Y);

        /// @brief Copies elements from one vector to another, wrapper to BLAS function scopy.
        ///
        /// Performs Y = X where X and Y are vectors of size nelemXY.
        ///
        /// @param nelemXY Number of elements in vectors X and Y.
        /// @param X Pointer to the first element of source vector X (size nelemXY), stride 1 assumed.
        /// @param Y Pointer to the first element of destination vector Y (size nelemXY), stride 1 assumed.
        void CopyVectors(const size_t nelemXY, const float *X, float *Y);

        /// @brief Copies elements from one vector to another with strides, wrapper to BLAS function dcopy.
        ///
        /// Performs Y = X where X and Y are vectors of size nelemXY with specified strides.
        ///
        /// @param nelemXY Number of elements in vectors X and Y.
        /// @param X Pointer to the first element of source vector X (size nelemXY).
        /// @param strideX Stride between consecutive elements in vector X.
        /// @param Y Pointer to the first element of destination vector Y (size nelemXY).
        /// @param strideY Stride between consecutive elements in vector Y.
        void CopyVectors(const size_t nelemXY, const double *X, const size_t strideX, double *Y, const size_t strideY);

        /// @brief Copies elements from one vector to another with strides, wrapper to BLAS function scopy.
        ///
        /// Performs Y = X where X and Y are vectors of size nelemXY with specified strides.
        ///
        /// @param nelemXY Number of elements in vectors X and Y.
        /// @param X Pointer to the first element of source vector X (size nelemXY).
        /// @param strideX Stride between consecutive elements in vector X.
        /// @param Y Pointer to the first element of destination vector Y (size nelemXY).
        /// @param strideY Stride between consecutive elements in vector Y.
        void CopyVectors(const size_t nelemXY, const float *X, const size_t strideX, float *Y, const size_t strideY);

        /// @brief Copies elements from one vector to another with mixed precision, wrapper to dcopy and conversion.
        ///
        /// Performs Y = X (with type conversion) where X is double-precision and Y is single-precision, both of size nelemXY.
        ///
        /// @param nelemXY Number of elements in vectors X and Y.
        /// @param X Pointer to the first element of source vector X (size nelemXY, double-precision), stride 1 assumed.
        /// @param Y Pointer to the first element of destination vector Y (size nelemXY, single-precision), stride 1 assumed.
        void CopyVectors(const size_t nelemXY, const double *X, float *Y);

        /// @brief Copies elements from one vector to another with mixed precision, wrapper to scopy and conversion.
        ///
        /// Performs Y = X (with type conversion) where X is single-precision and Y is double-precision, both of size nelemXY.
        ///
        /// @param nelemXY Number of elements in vectors X and Y.
        /// @param X Pointer to the first element of source vector X (size nelemXY, single-precision), stride 1 assumed.
        /// @param Y Pointer to the first element of destination vector Y (size nelemXY, double-precision), stride 1 assumed.
        void CopyVectors(const size_t nelemXY, const float *X, double *Y);

        /// @brief Swaps elements between two vectors, wrapper to BLAS function dswap.
        ///
        /// Exchanges elements between vectors X and Y of size nelemXY.
        ///
        /// @param nelemXY Number of elements in vectors X and Y.
        /// @param X Pointer to the first element of vector X (size nelemXY), stride 1 assumed.
        /// @param Y Pointer to the first element of vector Y (size nelemXY), stride 1 assumed.
        void SwapVectors(const size_t nelemXY, double *X, double *Y);

        /// @brief Swaps elements between two vectors, wrapper to BLAS function sswap.
        ///
        /// Exchanges elements between vectors X and Y of size nelemXY.
        ///
        /// @param nelemXY Number of elements in vectors X and Y.
        /// @param X Pointer to the first element of vector X (size nelemXY), stride 1 assumed.
        /// @param Y Pointer to the first element of vector Y (size nelemXY), stride 1 assumed.
        void SwapVectors(const size_t nelemXY, float *X, float *Y);

        /// @brief Swaps elements between two vectors with strides, wrapper to BLAS function dswap.
        ///
        /// Exchanges elements between vectors X and Y of size nelemXY with specified strides.
        ///
        /// @param nelemXY Number of elements in vectors X and Y.
        /// @param X Pointer to the first element of vector X (size nelemXY).
        /// @param strideX Stride between consecutive elements in vector X.
        /// @param Y Pointer to the first element of vector Y (size nelemXY).
        /// @param strideY Stride between consecutive elements in vector Y.
        void SwapVectors(const size_t nelemXY, double *X, const size_t strideX, double *Y, const size_t strideY);

        /// @brief Swaps elements between two vectors with strides, wrapper to BLAS function sswap.
        ///
        /// Exchanges elements between vectors X and Y of size nelemXY with specified strides.
        ///
        /// @param nelemXY Number of elements in vectors X and Y.
        /// @param X Pointer to the first element of vector X (size nelemXY).
        /// @param strideX Stride between consecutive elements in vector X.
        /// @param Y Pointer to the first element of vector Y (size nelemXY).
        /// @param strideY Stride between consecutive elements in vector Y.
        void SwapVectors(const size_t nelemXY, float *X, const size_t strideX, float *Y, const size_t strideY);

        /// @brief Scales a vector by a constant, wrapper to BLAS function dscal.
        ///
        /// Performs x = a * x where x is a vector of size ndim.
        ///
        /// @param ndim Number of elements in vector x.
        /// @param a Scaling factor.
        /// @param x Pointer to the first element of vector x (size ndim), stride 1 assumed.
        void ScaleVector(const size_t ndim, const double a, double *x);

        /// @brief Scales a vector by a constant, wrapper to BLAS function sscal.
        ///
        /// Performs x = a * x where x is a vector of size ndim.
        ///
        /// @param ndim Number of elements in vector x.
        /// @param a Scaling factor.
        /// @param x Pointer to the first element of vector x (size ndim), stride 1 assumed.
        void ScaleVector(const size_t ndim, const float a, float *x);

        /// @brief Scales a vector by a constant with stride, wrapper to BLAS function dscal.
        ///
        /// Performs x = a * x where x is a vector of size ndim with specified stride.
        ///
        /// @param ndim Number of elements in vector x.
        /// @param a Scaling factor.
        /// @param x Pointer to the first element of vector x (size ndim).
        /// @param ix Stride between consecutive elements in vector x.
        void ScaleVector(const size_t ndim, const double a, double *x, size_t ix);

        /// @brief Scales a vector by a constant with stride, wrapper to BLAS function sscal.
        ///
        /// Performs x = a * x where x is a vector of size ndim with specified stride.
        ///
        /// @param ndim Number of elements in vector x.
        /// @param a Scaling factor.
        /// @param x Pointer to the first element of vector x (size ndim).
        /// @param ix Stride between consecutive elements in vector x.
        void ScaleVector(const size_t ndim, const float a, float *x, size_t ix);

        /// @brief Finds the index of the maximum element in a vector, wrapper to BLAS function idamax.
        ///
        /// Returns the index of the element with maximum absolute value in vector x of size ndim.
        ///
        /// @param ndim Number of elements in vector x.
        /// @param x Pointer to the first element of vector x (size ndim), stride 1 assumed.
        /// @return Index (0-based) of the maximum element in vector x.
        int IndexMaxFromVector(const size_t ndim, const double *x);

        /// @brief Finds the index of the maximum element in a vector, wrapper to BLAS function isamax.
        ///
        /// Returns the index of the element with maximum absolute value in vector x of size ndim.
        ///
        /// @param ndim Number of elements in vector x.
        /// @param x Pointer to the first element of vector x (size ndim), stride 1 assumed.
        /// @return Index (0-based) of the maximum element in vector x.
        int IndexMaxFromVector(const size_t ndim, const float *x);

        /// @brief Finds the index of the maximum element in a vector with stride, wrapper to BLAS function idamax.
        ///
        /// Returns the index of the element with maximum absolute value in vector x of size ndim with specified stride.
        ///
        /// @param ndim Number of elements in vector x.
        /// @param x Pointer to the first element of vector x (size ndim).
        /// @param ix Stride between consecutive elements in vector x.
        /// @return Index (0-based) of the maximum element in vector x.
        int IndexMaxFromVector(const size_t ndim, const double *x, const size_t ix);

        /// @brief Finds the index of the maximum element in a vector with stride, wrapper to BLAS function isamax.
        ///
        /// Returns the index of the element with maximum absolute value in vector x of size ndim with specified stride.
        ///
        /// @param ndim Number of elements in vector x.
        /// @param x Pointer to the first element of vector x (size ndim).
        /// @param ix Stride between consecutive elements in vector x.
        /// @return Index (0-based) of the maximum element in vector x.
        int IndexMaxFromVector(const size_t ndim, const float *x, const size_t ix);

    }
}

#endif