/// @file mp-runtime.hpp
/// @brief Mixed-precision GPU computation runtime configuration and settings.
///
/// Provides runtime settings for mixed-precision Ozaki-type computations on GPU

#include "const.h"
#include "runtime.hpp"

#include "tensor/gpu/matrix.hpp"
#include "tensor/allocators.hpp"

namespace lahva
{
    namespace gpu
    {
        /// @brief Multi-Precision runtime settings
        class MPRuntime
        {
        public:
            // default number of splits for multi-precision Ozaki-type computations
            /// @brief number of splits for FP64
            size_t nsplits_FP64 = 4;
            /// @brief number of splits for FP32
            size_t nsplits_FP32 = 2;
            /// @brief fast mode flag, uses only the upper triangle of the combination of splits
            bool fast_mode = true;
            /// @brief batch mode flag, uses batched kernels for mp computations
            bool batch_mode = false;
        protected:
            mutable Matrix<double, CudaHostAllocator<double>, CudaDeviceAsyncAllocator<double>> mp_buffer_fp64_;
            mutable Matrix<float, CudaHostAllocator<float>, CudaDeviceAsyncAllocator<float>> mp_buffer_fp32_;
        public:
            MPRuntime(){};
            ~MPRuntime(){};
            
            template <typename high>
            Matrix<high>& getMPBuffer(const CudaRuntime& cudart, const Shape& shape) const
            {
                if constexpr (std::is_same<high, double>::value)
                {
                    if (mp_buffer_fp64_.size() < (shape.first * shape.second))
                    {
                        mp_buffer_fp64_ = Matrix<double, CudaHostAllocator<double>, CudaDeviceAsyncAllocator<double>>(shape, cudart);
                    }
                    return reinterpret_cast<Matrix<high>&>(mp_buffer_fp64_);
                }
                else if constexpr (std::is_same<high, float>::value)
                {
                    if (mp_buffer_fp32_.size() < (shape.first * shape.second))
                    {
                        mp_buffer_fp32_ = Matrix<float, CudaHostAllocator<float>, CudaDeviceAsyncAllocator<float>>(shape, cudart);
                    }
                    return reinterpret_cast<Matrix<high>&>(mp_buffer_fp32_);
                }
            };
            
            template <typename high>
            size_t getMaxSplits() const
            {
                if constexpr (std::is_same<high, float>::value)
                {
                    return nsplits_FP32;
                }
                else if constexpr (std::is_same<high, double>::value)
                {
                    return nsplits_FP64;
                }
            }
        };
    } // namespace gpu
}