#include "linalg.hpp"
#include "runtime.hpp"
#include "impl/blas/gpu/additional-level1.hpp"
#include "impl/blas/gpu/level1.hpp"
#include "custom-kernel/hadamard.h"
#include "../gpu-utils/utils.hpp"
#include "add-vectors.hpp"
namespace lahva{
    namespace gpu
    {
        template<>
        void HadamardProduct<double>(const CudaRuntime& cudart, const GPUTensor_<double>& vecin, GPUTensor_<double>& vecinout)
        {   
            check_device_alloc(cudart, vecinout);
            check_device_alloc(cudart, vecin);
            dHadamard(cudart, vecinout.gpu_data(), vecin.gpu_data(), vecin.size());            
        }

        template<>
        void HadamardProduct<float>(const CudaRuntime& cudart, const GPUTensor_<float>& vecin, GPUTensor_<float>& vecinout)
        {   
            check_device_alloc(cudart, vecinout);
            check_device_alloc(cudart, vecin);

            sHadamard(cudart, vecinout.gpu_data(), vecin.gpu_data(), vecin.size());
            
        }


        template<>
        void HadamardProduct<double>(const CudaRuntime& cudart, const GPUTensor_<double>& vecin, const GPUTensor_<double>& vecin2, GPUTensor_<double>& vecout)
        {
            check_device_alloc(cudart, vecin2);
            check_device_alloc(cudart, vecin);
            check_device_alloc(cudart, vecout);

            dHadamardcopy(cudart, vecout.gpu_data(), vecin.gpu_data(), vecin2.gpu_data(), vecout.size());            
        }

        template<>
        void HadamardProduct<float>(const CudaRuntime& cudart, const GPUTensor_<float>& vecin, const GPUTensor_<float>& vecin2, GPUTensor_<float>& vecout)
        {
            check_device_alloc(cudart, vecin2);
            check_device_alloc(cudart, vecin);
            check_device_alloc(cudart, vecout);

            sHadamardcopy(cudart, vecout.gpu_data(), vecin.gpu_data(), vecin2.gpu_data(), vecout.size());            
        }

        template<>
        void HadamardProduct<double>(const CudaRuntime& cudart, const Matrix_<double>& vecin, const GPUTensor_<double>& vecin2, Matrix_<double>& vecout)
        {
            check_device_alloc(cudart, vecin2);
            check_device_alloc(cudart, vecin);
            check_device_alloc(cudart, vecout);

            dHadamardcopy(cudart, vecout.gpu_data(), vecin.gpu_data(), vecin2.gpu_data(), vecout.size(), vecin2.size());            
        }

        template<>
        void HadamardProduct<float>(const CudaRuntime& cudart, const Matrix_<float>& vecin, const GPUTensor_<float>& vecin2, Matrix_<float>& vecout)
        {
            check_device_alloc(cudart, vecin2);
            check_device_alloc(cudart, vecin);
            check_device_alloc(cudart, vecout);

            sHadamardcopy(cudart, vecout.gpu_data(), vecin.gpu_data(), vecin2.gpu_data(), vecout.size(), vecin2.size());            
        }

        void AddVectors(const CudaRuntime& cudart, double a, const GPUTensor_<double>& x, GPUTensor_<float>& y)
        {
            check_device_alloc(cudart, x);
            check_device_alloc(cudart, y);
            AddVector(cudart, x.size(), a, x, y);
        }
        void AddVectors(const CudaRuntime& cudart, double a, const GPUTensor_<float>& x, GPUTensor_<double>& y)
        {
            check_device_alloc(cudart, x);
            check_device_alloc(cudart, y);
            AddVector(cudart, x.size(), a, x, y);
        }

    } // namespace gpu
}   