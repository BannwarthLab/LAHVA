#include "cuda_runtime.h"
#include <cfloat>

namespace lahva
{
    namespace gpu
    {
        
        
        template <typename T>
        class max_
        {
        public:
            __device__ T operator()(T x, T y)
            {
                return x > y ? x : y;
            }
            __device__ T ini_value()
            {
                return -FLT_MAX;
            }
        };

        template <>
        class max_<float>
        {
        public:
            __device__ float operator()(float x, float y)
            {
                return fmaxf(x,y);
            }
            __device__ float ini_value()
            {
                return -FLT_MAX;
            }
        };

        template <>
        class max_<double>
        {
        public:
            __device__ double operator()(double x, double y)
            {
                return x > y ? x : y;
            }
            __device__ double ini_value()
            {
                return -DBL_MAX;
            }
        };


        template <typename T>
        class min_
        {
        public:
            __device__ T operator()(T x, T y)
            {
                return min(x,y);
            }
            __device__ T ini_value()
            {
                return FLT_MAX;
            }
        };

        template <>
        class min_<float>
        {
        public:
            __device__ float operator()(float x, float y)
            {
                return min(x,y);
            }
            __device__ float ini_value()
            {
                return FLT_MAX;
            }
        };

        template <>
        class min_<double>
        {
        public:
            __device__ double operator()(double x, double y)
            {
                return min(x,y);
            }
            __device__ double ini_value()
            {
                return DBL_MAX;
            }
        };


        template <typename T>
        class add_rn
        {
        public:
            __device__ T operator()(T x, T y)
            {
                return x + y;
            }
             __device__ T ini_value()
            {
                return static_cast<T>(0.0);
            }
        };

        template <>
        class add_rn<float>
        {
        public:
            __device__ float operator()(float x, float y)
            {
                return __fadd_rn(x, y);
            }
             __device__ float ini_value()
            {
                return static_cast<float>(0.0);
            }
        };

        template <>
        class add_rn<double>
        {
        public:
            __device__ double operator()(double x, double y)
            {
                return __dadd_rn(x, y);
            }
             __device__ double ini_value()
            {
                return static_cast<double>(0.0);
            }
        };

        template <typename T>
        class sub_rn
        {
        public:
            __device__ T operator()(T x, T y)
            {
                return x - y;
            }
        };

        template <>
        class sub_rn<float>
        {
        public:
            __device__ float operator()(float x, float y)
            {
                return __fsub_rn(x, y);
            }
        };

        template <>
        class sub_rn<double>
        {
        public:
            __device__ double operator()(double x, double y)
            {
                return __dsub_rn(x, y);
            }
        };

        template <typename T>
        class pow
        {
        public:
            __device__ T operator()(T x, T y)
            {
                return pow(x, y);
            }
        };
        
        template<typename T, class op>
        __global__ void ApplyKernel(unsigned long long ndim, const T *a, T* b, op func)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < ndim)
            {
                b[index] = func(a[index], b[index]);
            }
        
        };

        template<typename T, class op>
        __global__ void ApplyKernel(unsigned long long ndim, T *a, T b, op func)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < ndim)
            {
                a[index] = func(a[index], b[index]);
            }
        
        };
       
    } // namespace gpu

} // namespace lahva
