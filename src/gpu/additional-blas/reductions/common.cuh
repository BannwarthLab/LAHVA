#include "cuda_runtime.h"

namespace lahva
{
    namespace gpu
    {

        template <typename T>
        class add_rn
        {
        public:
            __device__ T operator()(T x, T y)
            {
                return x + y;
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
        };

        template <>
        class add_rn<double>
        {
        public:
            __device__ double operator()(double x, double y)
            {
                return __dadd_rn(x, y);
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
                a[index] = func(a[index], b);
            }
        
        };
       
    } // namespace gpu

} // namespace lahva
