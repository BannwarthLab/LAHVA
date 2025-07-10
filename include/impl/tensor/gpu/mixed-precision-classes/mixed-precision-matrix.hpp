#pragma once

#include "impl/tensor/allocators.hpp"
#include "impl/tensor/gpu/gputensor.hpp"
#include "impl/tensor/cpu/tensor.hpp"
#include "impl/tensor/cpu/matrix.hpp"
#include "impl/tensor/gpu/matrix.hpp"
#include <vector>
namespace lahva
{
    namespace gpu
    {

    
        template <typename high_prec, typename low_prec, class Allocator = CudaHostAllocator<high_prec>, class GPUAllocator = CudaDeviceAllocator<high_prec>>
        class MixedPrecisionMatrix : public  lahva::gpu::Matrix<high_prec, Allocator, GPUAllocator>
        {
            protected:
                using Matrix<high_prec, Allocator, GPUAllocator>::n_rows_;
                using Matrix<high_prec, Allocator, GPUAllocator>::n_cols_;
                using Matrix<high_prec, Allocator, GPUAllocator>::data_id_;
                using Matrix<high_prec, Allocator, GPUAllocator>::data_size_;
                using Matrix<high_prec, Allocator, GPUAllocator>::check_size_;
            public:  
                mutable int max_split_ = 0; // maximum number of splits for the mixed precision matrix
                mutable Matrix<high_prec, Allocator, GPUAllocator> work_buffer_; 
                mutable std::vector<Matrix<low_prec>> split_matrices_; // stores the low precision matrices
                mutable bool splitted_ = false; // indicates whether the matrix has been split
                mutable Vector<int> split_exponents_; // stores the indices for splitting the matrix

            public:
                
                using Matrix<high_prec, Allocator, GPUAllocator>::shape;
                
                
                MixedPrecisionMatrix(const Shape &shape, const Allocator &alloc = Allocator(), const GPUAllocator &gpualloc = GPUAllocator());
                MixedPrecisionMatrix(const Matrix<high_prec, Allocator, GPUAllocator> &other) : Matrix<high_prec, Allocator, GPUAllocator>(other),
                work_buffer_(other.shape()), splitted_(false)
                {
                }    

                void split(const CudaRuntime &cudart, int max_split) const;

                void merge(const CudaRuntime &cudart, const high_prec* alphas, high_prec ini_beta);

                inline size_t splitSize() const
                {
                    return split_matrices_.size();
                }

                int getSplitExponent(size_t i) const
                {
                    if (!splitted_)
                    {
                        throw std::runtime_error("Matrix has not been split yet.");
                    }
                    if (i >= split_exponents_.size())
                    {
                        throw std::out_of_range("Index out of range for split exponents.");
                    }
                    return split_exponents_[i];
                }

                const Matrix<low_prec>& getSplitMatrix(size_t i) const
                {
                    if (!splitted_)
                    {
                        throw std::runtime_error("Matrix has not been split yet.");
                    }
                    return split_matrices_[i];
                }

                std::vector<Matrix<low_prec>>& getSplitMatrices() const
                {
                    return split_matrices_;
                }

                void createSplitMatrices(const CudaRuntime &cudart, int maxsplit) const
                {
                    max_split_ = maxsplit;
                    if (split_matrices_.size() != (size_t)max_split_)
                    {
                        split_matrices_ = std::vector<Matrix<low_prec>>(max_split_, Matrix<low_prec>(this->shape(), cudart, this->get_gpuallocator()));
                        for (size_t i = 0; i < max_split_; ++i)
                        {
                            split_matrices_[i].allocateGPU(cudart);
                        }
                    }

                    if (split_exponents_.size() != (size_t)max_split_)
                    {
                        split_exponents_ = Vector<int>((size_t)max_split_, this->get_allocator(), this->get_gpuallocator());
                    }
                }

                ~MixedPrecisionMatrix() {};
            

            };
    } // namespace gpu
} // namespace lahva