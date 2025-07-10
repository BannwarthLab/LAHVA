#include "impl/tensor/gpu/mixed-precision-classes/mixed-precision-matrix.hpp"
#include "impl/tensor/allocators.hpp"
#include "impl/blas/gpu/additional-level1.hpp"
#include "impl/blas/gpu/level1.hpp"
#include "impl/tensor/gpu/matrix.hpp"
#include "impl/tensor/gpu/vector.hpp"
#include "runtime.hpp"
#include "timer.hpp"
namespace lahva
{
    namespace gpu
    {
        template <typename high, typename low, typename Allocator, typename GPUAllocator>
        MixedPrecisionMatrix<high, low, Allocator, GPUAllocator>::MixedPrecisionMatrix(const Shape &shape, const Allocator &alloc, const GPUAllocator &gpualloc)
            : Matrix<high, Allocator, GPUAllocator>(shape, alloc, gpualloc), work_buffer_(shape, alloc, gpualloc)
        {
        }

        template <typename high, typename low, typename Allocator, typename GPUAllocator>
        void MixedPrecisionMatrix<high, low, Allocator, GPUAllocator>::split(const CudaRuntime& cudart, int maxsplit) const 
        {
            
            
            this->createSplitMatrices(cudart, maxsplit);
            if constexpr (std::is_same<low, __half>::value)
            {
                
                //cudart.synchronize();
                CopyVectors(cudart, *this, work_buffer_);
                SplitMatrix<high, low>(cudart, work_buffer_, split_matrices_, split_exponents_, max_split_);
                splitted_ = true;
            }
        }

        template <typename high, typename low, typename Allocator, typename GPUAllocator>
        void MixedPrecisionMatrix<high, low, Allocator, GPUAllocator>::merge(const CudaRuntime &cudart, const high* alphas, high ini_beta = 1.0)
        {
            CPUTimer timer;
            timer.push("Scale");
            ScaleVector(cudart, ini_beta, *this);
            timer.pop();
            timer.push("Custom Sayxpy");
            for (size_t i = 0; i < split_matrices_.size(); ++i)
            {
                AddVectors(cudart, alphas[i], split_matrices_[i], *this);    
            }
            timer.pop();
            std::cout << timer.print_entries();
            
        }

        template class MixedPrecisionMatrix<float, __half, CudaHostAllocator<float>, CudaDeviceAllocator<float>>;
        template class MixedPrecisionMatrix<float, __half, CudaHostAllocator<float>, CudaDeviceAsyncAllocator<float>>;
        template class MixedPrecisionMatrix<float, float, CudaHostAllocator<float>, CudaDeviceAllocator<float>>;
        template class MixedPrecisionMatrix<float, float, CudaHostAllocator<float>, CudaDeviceAsyncAllocator<float>>;
        template class MixedPrecisionMatrix<double, __half, CudaHostAllocator<double>, CudaDeviceAllocator<double>>;
        template class MixedPrecisionMatrix<double, __half, CudaHostAllocator<double>, CudaDeviceAsyncAllocator<double>>;
        template class MixedPrecisionMatrix<double, float, CudaHostAllocator<double>, CudaDeviceAllocator<double>>;
        template class MixedPrecisionMatrix<double, float, CudaHostAllocator<double>, CudaDeviceAsyncAllocator<double>>;
    } // namespace gpu
} // namespace lahva
