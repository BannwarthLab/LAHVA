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
        template <typename high, typename Allocator, typename GPUAllocator>
        MixedPrecisionMatrix<high, Allocator, GPUAllocator>::MixedPrecisionMatrix(const Shape &shape, const Allocator &alloc, const GPUAllocator &gpualloc)
            : Matrix<high, Allocator, GPUAllocator>(shape, alloc, gpualloc)
        {
        }
        
        template <typename high, typename Allocator, typename GPUAllocator>
        template <typename split_type>
        void MixedPrecisionMatrix<high, Allocator, GPUAllocator>::split(const CudaRuntime& cudart, int maxsplit, Matrix_<high>& buffer) const 
        {
            if (not splitted_fp16_)
            {
                        
            this->template createSplitMatrices<split_type>(cudart, maxsplit);

            if constexpr (std::is_same<split_type, __half>::value)
            {
                
                CopyVectors(cudart, *this, buffer);
                SplitMatrix<high, __half>(cudart, buffer, split_matrices_fp16_, split_exponents_, max_split_);
                
            }
            }
            else if (not splitted_fp32_)
            {
                this->template createSplitMatrices<split_type>(cudart, maxsplit);
            }
        }

        template <typename high, typename Allocator, typename GPUAllocator>
        void MixedPrecisionMatrix<high, Allocator, GPUAllocator>::merge(const CudaRuntime &cudart, const high* alphas, high ini_beta)
        {
            ScaleVector(cudart, ini_beta, *this);
            
            
            for (size_t i = 0; i < max_split_ ; ++i)
            {    
                AddVectors(cudart, alphas[i], this->getSplitMatrix<float>(i), *this);    
            }
            
            
        }

        template <typename high, typename Allocator, typename GPUAllocator>
        void MixedPrecisionMatrix<high, Allocator, GPUAllocator>::merge(const CudaRuntime &cudart)
        {
            
            for (size_t i = 0; i < max_split_ ; ++i)
            {
                AddVectors(cudart, 1.0, this->getSplitMatrix<float>(i), *this);
            }
        }


        template class MixedPrecisionMatrix<float, CudaHostAllocator<float>, CudaDeviceAllocator<float>>;
        template class MixedPrecisionMatrix<float, CudaHostAllocator<float>, CudaDeviceAsyncAllocator<float>>;
        template class MixedPrecisionMatrix<double, CudaHostAllocator<double>, CudaDeviceAllocator<double>>;
        template class MixedPrecisionMatrix<double, CudaHostAllocator<double>, CudaDeviceAsyncAllocator<double>>;
        template class MixedPrecisionMatrix<float, StdAllocator<float>, CudaDeviceAllocator<float>>;
        template class MixedPrecisionMatrix<float, StdAllocator<float>, CudaDeviceAsyncAllocator<float>>;
        template class MixedPrecisionMatrix<double, StdAllocator<double>, CudaDeviceAllocator<double>>;
        template class MixedPrecisionMatrix<double, StdAllocator<double>, CudaDeviceAsyncAllocator<double>>;        

        template void MixedPrecisionMatrix<float, CudaHostAllocator<float>, CudaDeviceAllocator<float>>::split<__half>(const CudaRuntime&, int, Matrix_<float>&) const;
        template void MixedPrecisionMatrix<float, CudaHostAllocator<float>, CudaDeviceAsyncAllocator<float>>::split<__half>(const CudaRuntime&, int, Matrix_<float>&) const;
        template void MixedPrecisionMatrix<double, CudaHostAllocator<double>, CudaDeviceAllocator<double>>::split<__half>(const CudaRuntime&, int, Matrix_<double>&) const;
        template void MixedPrecisionMatrix<double, CudaHostAllocator<double>, CudaDeviceAsyncAllocator<double>>::split<__half>(const CudaRuntime&, int, Matrix_<double>&) const;

        template void MixedPrecisionMatrix<float, StdAllocator<float>, CudaDeviceAllocator<float>>::split<__half>(const CudaRuntime&, int, Matrix_<float>&) const;
        template void MixedPrecisionMatrix<float, StdAllocator<float>, CudaDeviceAsyncAllocator<float>>::split<__half>(const CudaRuntime&, int, Matrix_<float>&) const;
        template void MixedPrecisionMatrix<double, StdAllocator<double>, CudaDeviceAllocator<double>>::split<__half>(const CudaRuntime&, int, Matrix_<double>&) const;
        template void MixedPrecisionMatrix<double, StdAllocator<double>, CudaDeviceAsyncAllocator<double>>::split<__half>(const CudaRuntime&, int, Matrix_<double>&) const;

        template void MixedPrecisionMatrix<float, CudaHostAllocator<float>, CudaDeviceAllocator<float>>::split<float>(const CudaRuntime&, int, Matrix_<float>&) const;
        template void MixedPrecisionMatrix<float, CudaHostAllocator<float>, CudaDeviceAsyncAllocator<float>>::split<float>(const CudaRuntime&, int, Matrix_<float>&) const;
        template void MixedPrecisionMatrix<double, CudaHostAllocator<double>, CudaDeviceAllocator<double>>::split<float>(const CudaRuntime&, int, Matrix_<double>&) const;
        template void MixedPrecisionMatrix<double, CudaHostAllocator<double>, CudaDeviceAsyncAllocator<double>>::split<float>(const CudaRuntime&, int, Matrix_<double>&) const;

        template void MixedPrecisionMatrix<float, StdAllocator<float>, CudaDeviceAllocator<float>>::split<float>(const CudaRuntime&, int, Matrix_<float>&) const;
        template void MixedPrecisionMatrix<float, StdAllocator<float>, CudaDeviceAsyncAllocator<float>>::split<float>(const CudaRuntime&, int, Matrix_<float>&) const;
        template void MixedPrecisionMatrix<double, StdAllocator<double>, CudaDeviceAllocator<double>>::split<float>(const CudaRuntime&, int, Matrix_<double>&) const;
        template void MixedPrecisionMatrix<double, StdAllocator<double>, CudaDeviceAsyncAllocator<double>>::split<float>(const CudaRuntime&, int, Matrix_<double>&) const;
    } // namespace gpu
} // namespace lahva
