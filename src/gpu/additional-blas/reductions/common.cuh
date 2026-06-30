/// @file common.cuh
/// @brief GPU reduction operation functors for custom reduction kernels.
///
/// Provides functors implementing various reduction operations (max, min, add, subtract, power)
/// with correctly-rounded arithmetic for GPU reduction algorithms.

#include <cfloat>
#include "cuda_runtime.h"

namespace lahva
{
    namespace gpu
    {
        /// @brief Functor for absolute maximum reduction operation.
        ///
        /// Computes the element with largest absolute value: max(|x|, |y|).
        ///
        /// @tparam T Floating-point type.
        template <typename T>
        class amax_
        {
        public:
            /// @brief Combines two values using absolute maximum operation.
            __device__ T operator()(T x, T y)
            {
                return fabs(x) > fabs(y) ? x : y;
            }
            /// @brief Returns initial value for reduction (zero).
            __device__ T ini_value()
            {
                return 0.0;
            }
        };
        
        /// @brief Functor for maximum reduction operation (generic).
        ///
        /// Computes the larger element: max(x, y).
        ///
        /// @tparam T Floating-point type.
        template <typename T>
        class max_
        {
        public:
            /// @brief Combines two values using maximum operation.
            __device__ T operator()(T x, T y)
            {
                return x > y ? x : y;
            }
            /// @brief Returns initial value for reduction.
            __device__ T ini_value()
            {
                return -FLT_MAX;
            }
        };

        /// @brief Functor for maximum reduction operation (single-precision specialization).
        ///
        /// Specialized for float using hardware __fmaxf operation.
        template <>
        class max_<float>
        {
        public:
            /// @brief Combines two single-precision values using hardware maximum.
            __device__ float operator()(float x, float y)
            {
                return fmaxf(x, y);
            }
            /// @brief Returns initial value for reduction.
            __device__ float ini_value()
            {
                return -FLT_MAX;
            }
        };

        /// @brief Functor for maximum reduction operation (double-precision specialization).
        ///
        /// Specialized for double using comparison operator.
        template <>
        class max_<double>
        {
        public:
            /// @brief Combines two double-precision values using comparison.
            __device__ double operator()(double x, double y)
            {
                return x > y ? x : y;
            }
            /// @brief Returns initial value for reduction.
            __device__ double ini_value()
            {
                return -DBL_MAX;
            }
        };


        /// @brief Functor for minimum reduction operation (generic).
        ///
        /// Computes the smaller element: min(x, y).
        ///
        /// @tparam T Floating-point type.
        template <typename T>
        class min_
        {
        public:
            /// @brief Combines two values using minimum operation.
            __device__ T operator()(T x, T y)
            {
                return min(x, y);
            }
            /// @brief Returns initial value for reduction.
            __device__ T ini_value()
            {
                return FLT_MAX;
            }
        };

        /// @brief Functor for minimum reduction operation (single-precision specialization).
        ///
        /// Specialized for float using CUDA min function.
        template <>
        class min_<float>
        {
        public:
            /// @brief Combines two single-precision values using minimum.
            __device__ float operator()(float x, float y)
            {
                return min(x, y);
            }
            /// @brief Returns initial value for reduction.
            __device__ float ini_value()
            {
                return FLT_MAX;
            }
        };

        /// @brief Functor for minimum reduction operation (double-precision specialization).
        ///
        /// Specialized for double using CUDA min function.
        template <>
        class min_<double>
        {
        public:
            /// @brief Combines two double-precision values using minimum.
            __device__ double operator()(double x, double y)
            {
                return min(x, y);
            }
            /// @brief Returns initial value for reduction.
            __device__ double ini_value()
            {
                return DBL_MAX;
            }
        };


        /// @brief Functor for correctly-rounded addition reduction (generic).
        ///
        /// Computes sum with standard addition: x + y.
        ///
        /// @tparam T Floating-point type.
        template <typename T>
        class add_rn
        {
        public:
            /// @brief Combines two values using addition.
            __device__ T operator()(T x, T y)
            {
                return x + y;
            }
            /// @brief Returns initial value for reduction (zero).
            __device__ T ini_value()
            {
                return static_cast<T>(0.0);
            }
        };

        /// @brief Functor for correctly-rounded addition reduction (single-precision specialization).
        ///
        /// Specialized for float using hardware __fadd_rn (correctly-rounded addition).
        template <>
        class add_rn<float>
        {
        public:
            /// @brief Combines two single-precision values using correctly-rounded addition.
            __device__ float operator()(float x, float y)
            {
                return __fadd_rn(x, y);
            }
            /// @brief Returns initial value for reduction (zero).
            __device__ float ini_value()
            {
                return static_cast<float>(0.0);
            }
        };

        /// @brief Functor for correctly-rounded addition reduction (double-precision specialization).
        ///
        /// Specialized for double using hardware __dadd_rn (correctly-rounded addition).
        template <>
        class add_rn<double>
        {
        public:
            /// @brief Combines two double-precision values using correctly-rounded addition.
            __device__ double operator()(double x, double y)
            {
                return __dadd_rn(x, y);
            }
            /// @brief Returns initial value for reduction (zero).
            __device__ double ini_value()
            {
                return static_cast<double>(0.0);
            }
        };

        /// @brief Functor for correctly-rounded subtraction operation (generic).
        ///
        /// Computes difference with standard subtraction: x - y.
        ///
        /// @tparam T Floating-point type.
        template <typename T>
        class sub_rn
        {
        public:
            /// @brief Combines two values using subtraction.
            __device__ T operator()(T x, T y)
            {
                return x - y;
            }
        };

        /// @brief Functor for correctly-rounded subtraction operation (single-precision specialization).
        ///
        /// Specialized for float using hardware __fsub_rn (correctly-rounded subtraction).
        template <>
        class sub_rn<float>
        {
        public:
            /// @brief Computes correctly-rounded difference of two single-precision values.
            __device__ float operator()(float x, float y)
            {
                return __fsub_rn(x, y);
            }
        };

        /// @brief Functor for correctly-rounded subtraction operation (double-precision specialization).
        ///
        /// Specialized for double using hardware __dsub_rn (correctly-rounded subtraction).
        template <>
        class sub_rn<double>
        {
        public:
            /// @brief Computes correctly-rounded difference of two double-precision values.
            __device__ double operator()(double x, double y)
            {
                return __dsub_rn(x, y);
            }
        };

        /// @brief Functor for power operation.
        ///
        /// Computes x raised to power y: x^y.
        ///
        /// @tparam T Floating-point type.
        template <typename T>
        class pow
        {
        public:
            /// @brief Computes x raised to power y.
            __device__ T operator()(T x, T y)
            {
                return pow(x, y);
            }
        };
        
        /// @brief GPU kernel to apply binary operation element-wise between two arrays.
        ///
        /// Applies a binary operation func to each pair of elements:
        /// b[i] = func(a[i], b[i]).
        ///
        /// @tparam T Element type.
        /// @tparam op Binary operation functor type.
        /// @param ndim Total number of elements.
        /// @param a Input array (read-only).
        /// @param b Input/output array (updated with results).
        /// @param func Binary operation functor.
        template<typename T, class op>
        __global__ void ApplyKernel(unsigned long long ndim, const T *a, T* b, op func)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < ndim)
            {
                b[index] = func(a[index], b[index]);
            }
        };

        /// @brief GPU kernel to apply binary operation element-wise between array and scalar.
        ///
        /// Applies a binary operation func to each element with a scalar:
        /// a[i] = func(a[i], b).
        ///
        /// @tparam T Element type.
        /// @tparam op Binary operation functor type.
        /// @param ndim Total number of elements.
        /// @param a Input/output array (updated with results).
        /// @param b Scalar value.
        /// @param func Binary operation functor.
        template<typename T, class op>
        __global__ void ApplyKernel(unsigned long long ndim, T *a, T b, op func)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < ndim)
            {
                a[index] = func(a[index], b);
            }
        };
       
    } // namespace gpu
} // namespace lahva
