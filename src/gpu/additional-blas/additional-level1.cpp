/// @file additional-level1.cpp
/// @brief GPU implementations of additional Level-1 operations (element-wise operations).
///
/// Implements specialized GPU kernels for element-wise tensor operations including
/// Hadamard products (element-wise multiplication) and mixed-precision vector addition.

#include "../gpu-utils/utils.hpp"
#include "add-vectors.hpp"
#include "custom-kernel/hadamard.h"
#include "impl/blas/gpu/additional-level1.hpp"
#include "linalg.hpp"
#include "runtime.hpp"

namespace lahva{
    namespace gpu
    {
        /// @brief Computes in-place Hadamard product (element-wise multiplication) for double-precision tensors.
        ///
        /// Multiplies each element of vecinout by the corresponding element of vecin in-place.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param vecin Input double-precision tensor.
        /// @param vecinout Input/output double-precision tensor, replaced with element-wise product.
        template<>
        void HadamardProduct<double>(const CudaRuntime& cudart, const GPUTensor_<double>& vecin, GPUTensor_<double>& vecinout)
        {   
            check_device_alloc(cudart, vecinout);
            check_device_alloc(cudart, vecin);
            dHadamard(cudart, vecinout.gpu_data(), vecin.gpu_data(), vecin.size());            
        }

        /// @brief Computes in-place Hadamard product (element-wise multiplication) for single-precision tensors.
        ///
        /// Multiplies each element of vecinout by the corresponding element of vecin in-place.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param vecin Input single-precision tensor.
        /// @param vecinout Input/output single-precision tensor, replaced with element-wise product.
        template<>
        void HadamardProduct<float>(const CudaRuntime& cudart, const GPUTensor_<float>& vecin, GPUTensor_<float>& vecinout)
        {   
            check_device_alloc(cudart, vecinout);
            check_device_alloc(cudart, vecin);

            sHadamard(cudart, vecinout.gpu_data(), vecin.gpu_data(), vecin.size());
            
        }


        /// @brief Computes Hadamard product (element-wise multiplication) with separate output for double-precision tensors.
        ///
        /// Computes element-wise product of two input tensors and stores result in a separate output tensor.
        /// vecout = vecin * vecin2.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param vecin First input double-precision tensor.
        /// @param vecin2 Second input double-precision tensor.
        /// @param vecout Output double-precision tensor, replaced with element-wise product.
        template<>
        void HadamardProduct<double>(const CudaRuntime& cudart, const GPUTensor_<double>& vecin, const GPUTensor_<double>& vecin2, GPUTensor_<double>& vecout)
        {
            check_device_alloc(cudart, vecin2);
            check_device_alloc(cudart, vecin);
            check_device_alloc(cudart, vecout);

            dHadamard(cudart, vecout.gpu_data(), vecin.gpu_data(), vecin2.gpu_data(), vecout.size());            
        }

        /// @brief Computes Hadamard product (element-wise multiplication) with separate output for single-precision tensors.
        ///
        /// Computes element-wise product of two input tensors and stores result in a separate output tensor.
        /// vecout = vecin * vecin2.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param vecin First input single-precision tensor.
        /// @param vecin2 Second input single-precision tensor.
        /// @param vecout Output single-precision tensor, replaced with element-wise product.
        template<>
        void HadamardProduct<float>(const CudaRuntime& cudart, const GPUTensor_<float>& vecin, const GPUTensor_<float>& vecin2, GPUTensor_<float>& vecout)
        {
            check_device_alloc(cudart, vecin2);
            check_device_alloc(cudart, vecin);
            check_device_alloc(cudart, vecout);

            sHadamard(cudart, vecout.gpu_data(), vecin.gpu_data(), vecin2.gpu_data(), vecout.size());            
        }

        /// @brief Computes mixed-precision scaled vector addition: y = a*x + y (double to float).
        ///
        /// Performs scaled vector addition converting from double-precision input to single-precision output.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param a Scalar factor for vector x.
        /// @param x Input double-precision tensor.
        /// @param y Input/output single-precision tensor, replaced with result (converted to float).
        void AddVectors(const CudaRuntime& cudart, double a, const GPUTensor_<double>& x, GPUTensor_<float>& y)
        {
            check_device_alloc(cudart, x);
            check_device_alloc(cudart, y);
            AddVector(cudart, x.size(), a, x, y);
        }

        /// @brief Computes mixed-precision scaled vector addition: y = a*x + y (float to double).
        ///
        /// Performs scaled vector addition converting from single-precision input to double-precision output.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param a Scalar factor for vector x.
        /// @param x Input single-precision tensor.
        /// @param y Input/output double-precision tensor, replaced with result.
        void AddVectors(const CudaRuntime& cudart, double a, const GPUTensor_<float>& x, GPUTensor_<double>& y)
        {
            check_device_alloc(cudart, x);
            check_device_alloc(cudart, y);
            AddVector(cudart, x.size(), a, x, y);
        }

    } // namespace gpu
} // namespace lahva