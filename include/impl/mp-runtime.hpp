#include "const.h"
#include "runtime.hpp"

#include "tensor/gpu/matrix.hpp"
#include "tensor/allocators.hpp"

namespace lahva
{
    namespace gpu
    {
        
        /// @brief Mixed-Precision (MP) GPU runtime configuration and buffer management
        class MPRuntime
        {
        public:
            /// @brief Number of FP64 split blocks for mixed-precision operations
            size_t nsplits_FP64 = 4;

            /// @brief Number of FP32 split blocks for mixed-precision operations
            size_t nsplits_FP32 = 2;

            /// @brief Enable fast mode: use only upper triangle of split combinations
            bool fast_mode = true;

            /// @brief Enable batch mode: use batched GPU kernels for MP computations
            bool batch_mode = false;

        protected:
            /// @brief Reusable temporary matrix buffer for FP64 mixed-precision operations
            mutable Matrix<double, CudaHostAllocator<double>, CudaDeviceAsyncAllocator<double>> mp_buffer_fp64_;

            /// @brief Reusable temporary matrix buffer for FP32 mixed-precision operations
            mutable Matrix<float, CudaHostAllocator<float>, CudaDeviceAsyncAllocator<float>> mp_buffer_fp32_;

        public:
            /// @brief Default constructor with default mixed-precision settings
            MPRuntime(){};

            /// @brief Destructor - releases temporary buffers
            ~MPRuntime(){};

            /// @brief Get or allocate temporary buffer for mixed-precision operations
            /// @tparam high high-precision type (double or float)
            /// @param[in] cudart CUDA runtime for GPU allocation
            /// @param[in] shape desired matrix dimensions for buffer
            /// @return reference to FP64 or FP32 buffer matrix (depending on high_prec)
            /// @note Buffer lifetime extends to MPRuntime destruction
            template <typename high>
            Matrix<high>& getMPBuffer(const CudaRuntime& cudart, const Shape& shape) const
            {
                if constexpr (std::is_same<high, double>::value)
                {
                    /// Allocate or resize FP64 buffer if needed
                    if (mp_buffer_fp64_.size() < (shape.first * shape.second))
                    {
                        mp_buffer_fp64_ = Matrix<double, CudaHostAllocator<double>, CudaDeviceAsyncAllocator<double>>(shape, cudart);
                    }
                    return reinterpret_cast<Matrix<high>&>(mp_buffer_fp64_);
                }
                else if constexpr (std::is_same<high, float>::value)
                {
                    /// Allocate or resize FP32 buffer if needed
                    if (mp_buffer_fp32_.size() < (shape.first * shape.second))
                    {
                        mp_buffer_fp32_ = Matrix<float, CudaHostAllocator<float>, CudaDeviceAsyncAllocator<float>>(shape, cudart);
                    }
                    return reinterpret_cast<Matrix<high>&>(mp_buffer_fp32_);
                }
            };

            /// @brief Get maximum number of splits for given precision type
            /// @tparam high high-precision type (double or float)
            /// @return nsplits_FP32 if high==float, nsplits_FP64 if high==double
            template <typename high>
            size_t getMaxSplits() const
            {
                if constexpr (std::is_same<high, float>::value)
                {
                    /// Return FP32 split count for single-precision matrices
                    return nsplits_FP32;
                }
                else if constexpr (std::is_same<high, double>::value)
                {
                    /// Return FP64 split count for double-precision matrices
                    return nsplits_FP64;
                }
            }
        };

    } // namespace gpu
    
} // namespace lahva