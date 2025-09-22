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

        // -------------------------------------
        // constexpr functions
        // -------------------------------------

        // -------------------------------------
        // getEmin
        // -------------------------------------

        template <typename TYPE>
        constexpr int32_t getEmin()
        {
            //fprintf(OUTPUT, "OzBLAS error: TYPE is not specified in getEmin.\n");
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

        // -------------------------------------
        // getEpse
        // -------------------------------------
        template <typename TYPE>
        constexpr int32_t getEpse()
        {
            //fprintf(OUTPUT, "OzBLAS error: TYPE is not specified in getEpse.\n");
            exit(1);
            return 0;
        }
        template <>
constexpr int32_t getEpse <int32_t> () {
	return 31;
}
template <>
constexpr int32_t getEpse <__half> () {
	return 24;
}
template <>
constexpr int32_t getEpse <float> () {
	return 24;
}
template <>
constexpr int32_t getEpse <double> () {
	return 53;
}

        // -------------------------------------
        // getTypeMax
        // -------------------------------------
        template <typename TYPE>
        constexpr TYPE getTypeMax()
        {
            //fprintf(OUTPUT, "OzBLAS error: TYPE is not specified in getTypeMax.\n");
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

        // -------------------------------------
        // getTypeMin
        // -------------------------------------
        template <typename TYPE>
        constexpr TYPE getTypeMin()
        {
            //fprintf(OUTPUT, "OzBLAS error: TYPE is not specified in getTypeMin.\n");
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

        template <typename TYPE>
        __host__ __device__ constexpr int32_t getDig()
        {
            //fprintf(OUTPUT, "OzBLAS error: TYPE is not specified in getEpse.\n");
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

        

        template <typename T>
        __device__ inline T getSub(T val1, T val2)
        {
            return val1 + val2;
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