/// @file common.h
/// @brief Common numeric utility templates for GPU custom kernels.
///
/// Provides compile-time and runtime utilities for querying floating-point type properties,
/// performing rounded arithmetic operations, and managing precision-specific constants.

#pragma once
#include <cmath>
#include <typeinfo>
#include <float.h>

#include "runtime.hpp"
#include <cuda_fp16.h>

namespace lahva
{
    namespace gpu
    {
        /// @brief Returns the minimum exponent (Emin) for the given floating-point type.
        ///
        /// Specializations return type-specific minimum exponent values used for underflow detection.
        template <typename TYPE>
        constexpr int32_t getEmin()
        {
            exit(1);
            return 0;
        }
        template <>
        constexpr int32_t getEmin<float>()
        {
            return -126;
        }
        template <>
        constexpr int32_t getEmin<double>()
        {
            return -1022;
        }

        /// @brief Returns the significand precision (number of significant bits) for the given type.
        ///
        /// Specializations return type-specific bit precision.
        template <typename TYPE>
        constexpr int32_t getEpse()
        {
            exit(1);
            return 0;
        }
        template <>
        constexpr int32_t getEpse<int32_t>()
        {
            return 31;
        }
        template <>
        constexpr int32_t getEpse<__half>()
        {
            return 24;
        }
        template <>
        constexpr int32_t getEpse<float>()
        {
            return 24;
        }
        template <>
        constexpr int32_t getEpse<double>()
        {
            return 53;
        }

        /// @brief Returns the maximum representable value for the given floating-point type.
        template <typename TYPE>
        constexpr TYPE getTypeMax()
        {
            exit(1);
            return 0;
        }
        template <>
        constexpr float getTypeMax<float>()
        {
            return FLT_MAX;
        }
        template <>
        constexpr double getTypeMax<double>()
        {
            return DBL_MAX;
        }

        /// @brief Returns the minimum positive normal representable value for the given type.
        template <typename TYPE>
        constexpr TYPE getTypeMin()
        {
            exit(1);
            return 0;
        }
        template <>
        constexpr float getTypeMin<float>()
        {
            return FLT_MIN;
        }
        template <>
        constexpr double getTypeMin<double>()
        {
            return DBL_MIN;
        }

        /// @brief Returns the number of decimal digits for the given floating-point type.
        template <typename TYPE>
        __host__ __device__ constexpr int32_t getDig()
        {
            exit(1);
            return 0;
        }
        template <>
        __host__ __device__ constexpr int32_t getDig<float>()
        {
            return FLT_DIG;
        }
        template <>
        __host__ __device__ constexpr int32_t getDig<double>()
        {
            return DBL_DIG;
        }

        template <>
        __host__ __device__ constexpr int32_t getDig<__half>()
        {
            return 3;
        }

        /// @brief Performs rounded subtraction with type-specific operations.
        ///
        /// Uses hardware-supported rounded arithmetic for correct rounding in
        /// high-precision computations.
        template <typename T>
        __device__ inline T getSub(T val1, T val2)
        {
            return val1 - val2;
        }

        template <>
        __device__ inline double getSub(double val1, double val2)
        {
            return __dsub_rn(val1, val2);
        }

        template <>
        __device__ inline float getSub(float val1, float val2)
        {
            return __fsub_rn(val1, val2);
        }

        /// @brief Performs rounded addition with type-specific operations.
        ///
        /// Uses hardware-supported rounded arithmetic for correct rounding in
        /// high-precision computations.
        template <typename T>
        __device__ inline T getAdd(T val1, T val2)
        {
            return val1 + val2;
        }

        template <>
        __device__ inline double getAdd(double val1, double val2)
        {
            return __dadd_rn(val1, val2);
        }

        template <>
        __device__ inline float getAdd(float val1, float val2)
        {
            return __fadd_rn(val1, val2);
        }

        /// @brief Performs rounded fused multiply-add (FMA) with type-specific operations.
        ///
        /// Computes val1*val2+val3 with hardware support for correctly rounded FMA
        template <typename T>
        __device__ inline T getFMA(T val1, T val2, T val3)
        {
            return 0;
        }

        template <>
        __device__ inline double getFMA(double val1, double val2, double val3)
        {
            return __fma_rn(val1, val2, val3);
        }

        template <>
        __device__ inline float getFMA(float val1, float val2, float val3)
        {
            return __fmaf_rn(val1, val2, val3);
        }
    }
}