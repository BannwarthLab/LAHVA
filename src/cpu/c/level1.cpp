#include "impl/blas/cpu/level1.h"

namespace lahva
{
    namespace cpu
    {

        /// @brief Computes the inner product of two double-precision vectors, wrapper to BLAS function ddot.
        ///
        /// Performs result = X^T * Y where X and Y are vectors of size nelemXY.
        ///
        /// @param nelemXY Number of elements in vectors X and Y.
        /// @param X Pointer to the first element of vector X (size nelemXY), stride 1 assumed.
        /// @param Y Pointer to the first element of vector Y (size nelemXY), stride 1 assumed.
        /// @return The inner product of vectors X and Y.
        double InnerVectorProduct(const size_t nelemXY, const double *X, const double *Y)
        {
            if (nelemXY == 0)
                return 0.0;
            BLAS_INT nXY = (BLAS_INT)nelemXY;
            BLAS_INT incx = 1;
            BLAS_INT incy = 1;

            return cblas_ddot(nXY, X, incx, Y, incy);
        }

        /// @brief Computes the inner product of two single-precision vectors, wrapper to BLAS function sdot.
        ///
        /// Performs result = X^T * Y where X and Y are vectors of size nelemXY.
        ///
        /// @param nelemXY Number of elements in vectors X and Y.
        /// @param X Pointer to the first element of vector X (size nelemXY), stride 1 assumed.
        /// @param Y Pointer to the first element of vector Y (size nelemXY), stride 1 assumed.
        /// @return The inner product of vectors X and Y.
        float InnerVectorProduct(const size_t nelemXY, const float *X, const float *Y)
        {
            if (nelemXY == 0)
                return 0.0;
            BLAS_INT nXY = (BLAS_INT)nelemXY;
            BLAS_INT incx = 1;
            BLAS_INT incy = 1;

            return cblas_sdot(nXY, X, incx, Y, incy);
        }

        /// @brief Computes the inner product of two double-precision vectors with strides, wrapper to BLAS function ddot.
        ///
        /// Performs result = X^T * Y where X and Y are vectors of size nelemXY with specified strides.
        ///
        /// @param nelemXY Number of elements in vectors X and Y.
        /// @param X Pointer to the first element of vector X (size nelemXY).
        /// @param strideX Stride between consecutive elements in vector X.
        /// @param Y Pointer to the first element of vector Y (size nelemXY).
        /// @param strideY Stride between consecutive elements in vector Y.
        /// @return The inner product of vectors X and Y.
        double InnerVectorProduct(const size_t nelemXY, const double *X, const size_t strideX, const double *Y, const size_t strideY)
        {
            if (nelemXY == 0)
                return 0.0;
            BLAS_INT nXY = (BLAS_INT)nelemXY;
            BLAS_INT incx = (BLAS_INT)strideX;
            BLAS_INT incy = (BLAS_INT)strideY;
            return cblas_ddot(nXY, X, incx, Y, incy);
        }

        /// @brief Computes the inner product of two single-precision vectors with strides, wrapper to BLAS function sdot.
        ///
        /// Performs result = X^T * Y where X and Y are vectors of size nelemXY with specified strides.
        ///
        /// @param nelemXY Number of elements in vectors X and Y.
        /// @param X Pointer to the first element of vector X (size nelemXY).
        /// @param strideX Stride between consecutive elements in vector X.
        /// @param Y Pointer to the first element of vector Y (size nelemXY).
        /// @param strideY Stride between consecutive elements in vector Y.
        /// @return The inner product of vectors X and Y.
        float InnerVectorProduct(const size_t nelemXY, const float *X, const size_t strideX, const float *Y, const size_t strideY)
        {
            if (nelemXY == 0)
                return 0.0;
            BLAS_INT nXY = (BLAS_INT)nelemXY;
            BLAS_INT incx = (BLAS_INT)strideX;
            BLAS_INT incy = (BLAS_INT)strideY;

            return cblas_sdot(nXY, X, incx, Y, incy);
        }

        /// @brief Adds a scaled double-precision vector to another vector, wrapper to BLAS function daxpy.
        ///
        /// Performs y = y + a * x where x and y are vectors of size ndim.
        ///
        /// @param ndim Number of elements in vectors x and y.
        /// @param a Scalar multiplier for vector x.
        /// @param x Pointer to the first element of vector x (size ndim), stride 1 assumed.
        /// @param y Pointer to the first element of vector y (size ndim), stride 1 assumed.
        void AddVectors(const size_t ndim, const double a, const double *x, double *y)
        {
            const BLAS_INT n = (BLAS_INT)ndim;
            const BLAS_INT one = 1;
            cblas_daxpy(n, a, x, one, y, one);
        }

        /// @brief Adds a scaled single-precision vector to another vector, wrapper to BLAS function saxpy.
        ///
        /// Performs y = y + a * x where x and y are vectors of size ndim.
        ///
        /// @param ndim Number of elements in vectors x and y.
        /// @param a Scalar multiplier for vector x.
        /// @param x Pointer to the first element of vector x (size ndim), stride 1 assumed.
        /// @param y Pointer to the first element of vector y (size ndim), stride 1 assumed.
        void AddVectors(const size_t ndim, const float a, const float *x, float *y)
        {
            const BLAS_INT n = (BLAS_INT)ndim;
            const BLAS_INT one = 1;
            cblas_saxpy(n, a, x, one, y, one);
        }

        /// @brief Adds a scaled double-precision vector to another vector with strides, wrapper to BLAS function daxpy.
        ///
        /// Performs y = y + a * x where x and y are vectors of size ndim with specified strides.
        ///
        /// @param ndim Number of elements in vectors x and y.
        /// @param a Scalar multiplier for vector x.
        /// @param x Pointer to the first element of vector x (size ndim).
        /// @param ix Stride between consecutive elements in vector x.
        /// @param y Pointer to the first element of vector y (size ndim).
        /// @param iy Stride between consecutive elements in vector y.
        void AddVectors(const size_t ndim, const double a, const double *x, size_t ix, double *y, size_t iy)
        {
            const BLAS_INT n = (BLAS_INT)ndim;
            const BLAS_INT incx = (BLAS_INT)ix;
            const BLAS_INT incy = (BLAS_INT)iy;
            cblas_daxpy(n, a, x, incx, y, incy);
        }

        /// @brief Adds a scaled single-precision vector to another vector with strides, wrapper to BLAS function saxpy.
        ///
        /// Performs y = y + a * x where x and y are vectors of size ndim with specified strides.
        ///
        /// @param ndim Number of elements in vectors x and y.
        /// @param a Scalar multiplier for vector x.
        /// @param x Pointer to the first element of vector x (size ndim).
        /// @param ix Stride between consecutive elements in vector x.
        /// @param y Pointer to the first element of vector y (size ndim).
        /// @param iy Stride between consecutive elements in vector y.
        void AddVectors(const size_t ndim, const float a, const float *x, size_t ix, float *y, size_t iy)
        {
            const BLAS_INT n = (BLAS_INT)ndim;
            const BLAS_INT incx = (BLAS_INT)ix;
            const BLAS_INT incy = (BLAS_INT)iy;
            cblas_saxpy(n, a, x, incx, y, incy);
        }

        // Copy routines////////////////////////////////////////////////////////////////////

        /// @brief Copies elements from one double-precision vector to another, wrapper to BLAS function dcopy.
        ///
        /// Performs y = x where x and y are vectors of size ndim.
        ///
        /// @param ndim Number of elements in vectors x and y.
        /// @param x Pointer to the first element of vector x (size ndim), stride 1 assumed.
        /// @param y Pointer to the first element of vector y (size ndim), stride 1 assumed.
        void CopyVectors(const size_t ndim, const double *x, double *y)
        {
            const BLAS_INT n = (BLAS_INT)ndim;
            const BLAS_INT one = 1;
            cblas_dcopy(n, x, one, y, one);
        }

        /// @brief Copies elements from one single-precision vector to another, wrapper to BLAS function scopy.
        ///
        /// Performs y = x where x and y are vectors of size ndim.
        ///
        /// @param ndim Number of elements in vectors x and y.
        /// @param x Pointer to the first element of vector x (size ndim), stride 1 assumed.
        /// @param y Pointer to the first element of vector y (size ndim), stride 1 assumed.
        void CopyVectors(const size_t ndim, const float *x, float *y)
        {
            const BLAS_INT n = (BLAS_INT)ndim;
            const BLAS_INT one = 1;
            cblas_scopy(n, x, one, y, one);
        }

        /// @brief Copies elements from one double-precision vector to another with strides, wrapper to BLAS function dcopy.
        ///
        /// Performs y = x where x and y are vectors of size ndim with specified strides.
        ///
        /// @param ndim Number of elements in vectors x and y.
        /// @param x Pointer to the first element of vector x (size ndim).
        /// @param ix Stride between consecutive elements in vector x.
        /// @param y Pointer to the first element of vector y (size ndim).
        /// @param iy Stride between consecutive elements in vector y.
        void CopyVectors(const size_t ndim, const double *x, size_t ix, double *y, size_t iy)
        {
            const BLAS_INT n = (BLAS_INT)ndim;
            const BLAS_INT incx = (BLAS_INT)ix;
            const BLAS_INT incy = (BLAS_INT)iy;
            cblas_dcopy(n, x, incx, y, incy);
        }

        /// @brief Copies elements from one single-precision vector to another with strides, wrapper to BLAS function scopy.
        ///
        /// Performs y = x where x and y are vectors of size ndim with specified strides.
        ///
        /// @param ndim Number of elements in vectors x and y.
        /// @param x Pointer to the first element of vector x (size ndim).
        /// @param ix Stride between consecutive elements in vector x.
        /// @param y Pointer to the first element of vector y (size ndim).
        /// @param iy Stride between consecutive elements in vector y.
        void CopyVectors(const size_t ndim, const float *x, size_t ix, float *y, size_t iy)
        {
            const BLAS_INT n = (BLAS_INT)ndim;
            const BLAS_INT incx = (BLAS_INT)ix;
            const BLAS_INT incy = (BLAS_INT)iy;
            cblas_scopy(n, x, incx, y, incy);
        }

        /// @brief Copies a double-precision vector to a single-precision vector.
        ///
        /// Performs element-wise type conversion: Y[i] = (float)X[i].
        ///
        /// @param nelemXY Number of elements in both vectors.
        /// @param X Pointer to the source double-precision vector.
        /// @param Y Pointer to the destination single-precision vector.
        void CopyVectors(const size_t nelemXY, const double *X, float *Y)
        {

            // #pragma omp for
            for (size_t i = 0; i < nelemXY; i++)
            {
                Y[i] = static_cast<float>(X[i]);
            }
        }

        /// @brief Copies a single-precision vector to a double-precision vector.
        ///
        /// Performs element-wise type conversion: Y[i] = (double)X[i].
        ///
        /// @param nelemXY Number of elements in both vectors.
        /// @param X Pointer to the source single-precision vector.
        /// @param Y Pointer to the destination double-precision vector.
        void CopyVectors(const size_t nelemXY, const float *X, double *Y)
        {

            // #pragma omp for
            for (size_t i = 0; i < nelemXY; i++)
            {
                Y[i] = static_cast<double>(X[i]);
            }
        }

        // Swap routines////////////////////////////////////////////////////////////////////

        /// @brief Swaps two double-precision vectors, wrapper to BLAS function dswap.
        ///
        /// Performs x <-> y (element-wise swap) where x and y are vectors of size ndim.
        ///
        /// @param ndim Number of elements in vectors x and y.
        /// @param x Pointer to the first element of vector x (size ndim), stride 1 assumed.
        /// @param y Pointer to the first element of vector y (size ndim), stride 1 assumed.
        void SwapVectors(const size_t ndim, double *x, double *y)
        {
            const BLAS_INT n = (BLAS_INT)ndim;
            const BLAS_INT one = 1;
            cblas_dswap(n, x, one, y, one);
        }

        /// @brief Swaps two single-precision vectors, wrapper to BLAS function sswap.
        ///
        /// Performs x <-> y (element-wise swap) where x and y are vectors of size ndim.
        ///
        /// @param ndim Number of elements in vectors x and y.
        /// @param x Pointer to the first element of vector x (size ndim), stride 1 assumed.
        /// @param y Pointer to the first element of vector y (size ndim), stride 1 assumed.
        void SwapVectors(const size_t ndim, float *x, float *y)
        {
            const BLAS_INT n = (BLAS_INT)ndim;
            const BLAS_INT one = 1;
            cblas_sswap(n, x, one, y, one);
        }

        /// @brief Swaps two double-precision vectors with strides, wrapper to BLAS function dswap.
        ///
        /// Performs x <-> y (element-wise swap) where x and y are vectors of size ndim with specified strides.
        ///
        /// @param ndim Number of elements in vectors x and y.
        /// @param x Pointer to the first element of vector x (size ndim).
        /// @param ix Stride between consecutive elements in vector x.
        /// @param y Pointer to the first element of vector y (size ndim).
        /// @param iy Stride between consecutive elements in vector y.
        void SwapVectors(const size_t ndim, double *x, size_t ix, double *y, size_t iy)
        {
            const BLAS_INT n = (BLAS_INT)ndim;
            const BLAS_INT incx = (BLAS_INT)ix;
            const BLAS_INT incy = (BLAS_INT)iy;
            cblas_dswap(n, x, incx, y, incy);
        }

        /// @brief Swaps two single-precision vectors with strides, wrapper to BLAS function sswap.
        ///
        /// Performs x <-> y (element-wise swap) where x and y are vectors of size ndim with specified strides.
        ///
        /// @param ndim Number of elements in vectors x and y.
        /// @param x Pointer to the first element of vector x (size ndim).
        /// @param ix Stride between consecutive elements in vector x.
        /// @param y Pointer to the first element of vector y (size ndim).
        /// @param iy Stride between consecutive elements in vector y.
        void SwapVectors(const size_t ndim, float *x, size_t ix, float *y, size_t iy)
        {
            const BLAS_INT n = (BLAS_INT)ndim;
            const BLAS_INT incx = (BLAS_INT)ix;
            const BLAS_INT incy = (BLAS_INT)iy;
            cblas_sswap(n, x, incx, y, incy);
        }

        // Scale routines////////////////////////////////////////////////////////////////////

        /// @brief Scales a double-precision vector by a scalar multiplier, wrapper to BLAS function dscal.
        ///
        /// Performs x = a * x where x is a vector of size ndim.
        ///
        /// @param ndim Number of elements in vector x.
        /// @param a Scalar multiplier for vector x.
        /// @param x Pointer to the first element of vector x (size ndim), stride 1 assumed.
        void ScaleVector(const size_t ndim, const double a, double *x)
        {
            const BLAS_INT n = (BLAS_INT)ndim;
            const BLAS_INT one = 1;
            cblas_dscal(n, a, x, one);
        }
        /// @brief Scales a single-precision vector by a scalar multiplier, wrapper to BLAS function sscal.
        ///
        /// Performs x = a * x where x is a vector of size ndim.
        ///
        /// @param ndim Number of elements in vector x.
        /// @param a Scalar multiplier for vector x.
        /// @param x Pointer to the first element of vector x (size ndim), stride 1 assumed.
        void ScaleVector(const size_t ndim, const float a, float *x)
        {
            const BLAS_INT n = (BLAS_INT)ndim;
            const BLAS_INT one = 1;
            cblas_sscal(n, a, x, one);
        }

        /// @brief Scales a double-precision vector by a scalar multiplier with stride, wrapper to BLAS function dscal.
        ///
        /// Performs x = a * x where x is a vector of size ndim with specified stride.
        ///
        /// @param ndim Number of elements in vector x.
        /// @param a Scalar multiplier for vector x.
        /// @param x Pointer to the first element of vector x (size ndim).
        /// @param ix Stride between consecutive elements in vector x.
        void ScaleVector(const size_t ndim, const double a, double *x, size_t ix)
        {
            const BLAS_INT n = (BLAS_INT)ndim;
            const BLAS_INT incx = (BLAS_INT)ix;
            cblas_dscal(n, a, x, incx);
        }
        /// @brief Scales a single-precision vector by a scalar multiplier with stride, wrapper to BLAS function sscal.
        ///
        /// Performs x = a * x where x is a vector of size ndim with specified stride.
        ///
        /// @param ndim Number of elements in vector x.
        /// @param a Scalar multiplier for vector x.
        /// @param x Pointer to the first element of vector x (size ndim).
        /// @param ix Stride between consecutive elements in vector x.
        void ScaleVector(const size_t ndim, const float a, float *x, size_t ix)
        {
            const BLAS_INT n = (BLAS_INT)ndim;
            const BLAS_INT incx = (BLAS_INT)ix;
            cblas_sscal(n, a, x, incx);
        }

        /// @brief Finds the index of the element with the maximum absolute value in a double-precision vector, wrapper to BLAS function idamax.
        ///
        /// Returns the index of the element with the maximum absolute value in vector x.
        ///
        /// @param ndim Number of elements in vector x.
        /// @param x Pointer to the first element of vector x (size ndim), stride 1 assumed.
        /// @return Index (1-based) of the element with the maximum absolute value in x.
        int IndexMaxFromVector(const size_t ndim, const double *x)
        {
            const BLAS_INT n = (BLAS_INT)ndim;
            const BLAS_INT one = 1;

            return cblas_idamax(n, x, one);
        }
        /// @brief Finds the index of the element with the maximum absolute value in a single-precision vector, wrapper to BLAS function isamax.
        ///
        /// Returns the index of the element with the maximum absolute value in vector x.
        ///
        /// @param ndim Number of elements in vector x.
        /// @param x Pointer to the first element of vector x (size ndim), stride 1 assumed.
        /// @return Index (1-based) of the element with the maximum absolute value in x.
        int IndexMaxFromVector(const size_t ndim, const float *x)
        {
            const BLAS_INT n = (BLAS_INT)ndim;
            const BLAS_INT one = 1;

            return cblas_isamax(n, x, one);
        }
        /// @brief Finds the index of the element with the maximum absolute value in a double-precision vector with stride, wrapper to BLAS function idamax.
        ///
        /// Returns the index of the element with the maximum absolute value in vector x.
        ///
        /// @param ndim Number of elements in vector x.
        /// @param x Pointer to the first element of vector x (size ndim).
        /// @param ix Stride between consecutive elements in vector x.
        /// @return Index (1-based) of the element with the maximum absolute value in x.
        int IndexMaxFromVector(const size_t ndim, const double *x, size_t ix)
        {
            const BLAS_INT n = (BLAS_INT)ndim;
            const BLAS_INT incx = (BLAS_INT)ix;

            return cblas_idamax(n, x, incx);
        }
        /// @brief Finds the index of the element with the maximum absolute value in a single-precision vector with stride, wrapper to BLAS function isamax.
        ///
        /// Returns the index of the element with the maximum absolute value in vector x.
        ///
        /// @param ndim Number of elements in vector x.
        /// @param x Pointer to the first element of vector x (size ndim).
        /// @param ix Stride between consecutive elements in vector x.
        /// @return Index (1-based) of the element with the maximum absolute value in x.
        int IndexMaxFromVector(const size_t ndim, const float *x, size_t ix)
        {
            const BLAS_INT n = (BLAS_INT)ndim;
            const BLAS_INT incx = (BLAS_INT)ix;

            return cblas_isamax(n, x, incx);
        }

    }
}
