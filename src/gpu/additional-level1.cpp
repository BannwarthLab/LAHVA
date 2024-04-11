#include "linalg.hpp"
#include "runtime.hpp"
#include "additional-level1.hpp"
#include "custom-kernel/hadamard.h"
#include "../gpu-utils/utils.hpp"

namespace tcgmtensor{
    namespace gpu
    {
        template<>
        void Hadamard<double>(const CudaRuntime& cudart, const GPUTensor<double>& vecin, GPUTensor<double>& vecinout)
        {   
            check_device_alloc(cudart, vecinout);
            check_device_alloc(cudart, vecin);
            
            dHadamard(vecinout.gpu_data(), vecin.gpu_data(), vecinout.size());            
        }

        template<>
        void Hadamard<float>(const CudaRuntime& cudart, const GPUTensor<float>& vecin, GPUTensor<float>& vecinout)
        {   
            check_device_alloc(cudart, vecinout);
            check_device_alloc(cudart, vecin);

            sHadamard(vecinout.gpu_data(), vecin.gpu_data(), vecinout.size());
            
        }


        template<>
        void Hadamard<double>(const CudaRuntime& cudart, const GPUTensor<double>& vecin, const GPUTensor<double>& vecin2, GPUTensor<double>& vecout)
        {
            check_device_alloc(cudart, vecin2);
            check_device_alloc(cudart, vecin);
            check_device_alloc(cudart, vecout);


            dHadamardcopy(vecout.gpu_data(), vecin.gpu_data(), vecin2.gpu_data(), vecout.size());            
        }

        template<>
        void Hadamard<float>(const CudaRuntime& cudart, const GPUTensor<float>& vecin, const GPUTensor<float>& vecin2, GPUTensor<float>& vecout)
        {
            check_device_alloc(cudart, vecin2);
            check_device_alloc(cudart, vecin);
            check_device_alloc(cudart, vecout);


            sHadamardcopy(vecout.gpu_data(), vecin.gpu_data(), vecin2.gpu_data(), vecout.size());            
        }


    } // namespace gpu
}   