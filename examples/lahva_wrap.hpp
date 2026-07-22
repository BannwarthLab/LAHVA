#pragma once
#include <linalg.hpp>
#include <lahva.hpp>
#include <lahva.h>
#include <timer.hpp>
#ifdef _CUDA
#include <runtime.hpp>
#endif
namespace lahva
{
    
#ifdef _CUDA   
    namespace gpu
    {
        struct gpu_BLAS{
    
        };
    };
#endif
    namespace cpu
    {
        struct cpu_BLAS{
            
        };    
    };

#ifdef _CUDA
    template<typename blas_impl>
    class TensorFactory
    {
        public: 
            template<typename T>
            using CPUAll = CudaHostAllocator<T>;
            template<typename T>
            using GPUAll = CudaDeviceAsyncAllocator<T>;

            template<typename T>
            using Vector = typename std::conditional<std::is_same<blas_impl, cpuBLAS::cpu_BLAS>::value, cpu::Vector<T>, gpu::Vector<T,CPUAll<T>,GPUAll<T>>>::type;
            template<typename T>
            using Matrix = typename std::conditional<std::is_same<blas_impl, cpuBLAS::cpu_BLAS>::value, cpu::Matrix<T>, gpu::Matrix<T,CPUAll<T>,GPUAll<T>>>::type;
            template<typename T>
            using LowTriMatrix = typename std::conditional<std::is_same<blas_impl, cpuBLAS::cpu_BLAS>::value, cpu::LowTriMatrix<T>, gpu::LowTriMatrix<T,CPUAll<T>,GPUAll<T>>>::type;
            using Timer = typename std::conditional<std::is_same<blas_impl, cpuBLAS::cpu_BLAS>::value, CPUTimer, GPUTimer>::type;
            using RunTime = typename std::conditional<std::is_same<blas_impl, cpuBLAS::cpu_BLAS>::value, CPURuntime, CudaRuntime>::type;
    };
#else

template<typename blas_impl>
class TensorFactory
{
    public: 
        template<typename T>
        using CPUAll = StdAllocator<T>;


        template<typename T>
        using Vector = typename cpu::Vector<T, CPUAll<T>>;
        template<typename T>
        using Matrix = typename cpu::Matrix<T, CPUAll<T>>;
        template<typename T>
        using LowTriMatrix = typename cpu::LowTriMatrix<T, CPUAll<T>>;
        using Timer = CPUTimer;
        using RunTime = CPURuntime;
};
#endif
           
} // namespace lahva