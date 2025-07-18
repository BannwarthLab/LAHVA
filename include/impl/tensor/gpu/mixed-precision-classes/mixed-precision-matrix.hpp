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

        template <typename high_prec, class Allocator = CudaHostAllocator<high_prec>, class GPUAllocator = CudaDeviceAllocator<high_prec>>
        class MixedPrecisionMatrix : public lahva::gpu::Matrix<high_prec, Allocator, GPUAllocator>
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
            mutable std::vector<Matrix<__half>> split_matrices_fp16_; // stores the low precision matrices
            mutable std::vector<Matrix<float>> split_matrices_fp32_;
            mutable bool splitted_ = false;       // indicates whether the matrix has been split
            mutable Vector<int> split_exponents_; // stores the indices for splitting the matrix

        public:
            using Matrix<high_prec, Allocator, GPUAllocator>::shape;

            MixedPrecisionMatrix(const Shape &shape, const Allocator &alloc = Allocator(), const GPUAllocator &gpualloc = GPUAllocator());
            MixedPrecisionMatrix(const Matrix<high_prec, Allocator, GPUAllocator> &other) : 
            Matrix<high_prec, Allocator, GPUAllocator>(other),
            work_buffer_(other.shape()), splitted_(false)
            {
            }
            MixedPrecisionMatrix(const Shape &shape, std::initializer_list<high_prec> init, bool row_major = false, const Allocator &alloc = Allocator(), const GPUAllocator &gpualloc = GPUAllocator())
                : Matrix<high_prec, Allocator, GPUAllocator>(shape, init, row_major, alloc, gpualloc)
            {
            }
            
            template <typename split_type>
            void split(const CudaRuntime &cudart, int max_split) const;

            void merge(const CudaRuntime &cudart, const high_prec *alphas, high_prec ini_beta);

            template <typename split_type>
            inline size_t splitSize() const
            {
                if constexpr (std::is_same<split_type, __half>::value)
                {
                    return split_matrices_fp16_.size();
                }
                else if constexpr (std::is_same<split_type, float>::value)
                {
    
                    return split_matrices_fp32_.size();
                }
                else
                {
                    return 0;
                } 
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

            template <typename split_type>
            const Matrix<split_type> &getSplitMatrix(size_t i) const
            {
                if (!splitted_)
                {
                    throw std::runtime_error("Matrix has not been split yet.");
                }

                if constexpr (std::is_same<split_type, __half>::value)
                {
                    return split_matrices_fp16_[i];
                }
                else if constexpr (std::is_same<split_type, float>::value)
                {
                    return split_matrices_fp32_[i];
                } 
                
            }

            template <typename split_type>
            std::vector<Matrix<split_type>> &getSplitMatrices() const
            {
                if constexpr (std::is_same<split_type, __half>::value)
                {
                    return split_matrices_fp16_;
                }
                else if constexpr (std::is_same<split_type, float>::value)
                {
                    return split_matrices_fp32_;
                }
            }

            template <typename split_type>
            void createSplitMatrices(const CudaRuntime &cudart, int maxsplit) const
            {
                if constexpr (std::is_same<split_type, __half>::value)
                {
                    max_split_ = maxsplit;
                    if (split_matrices_fp16_.size() != (size_t)max_split_)
                    {
                        split_matrices_fp16_ = std::vector<Matrix<__half>>(max_split_, Matrix<__half>(this->shape(), cudart, this->get_gpuallocator()));
                        for (size_t i = 0; i < max_split_; ++i)
                        {
                            split_matrices_fp16_[i].allocateGPU(cudart);
                        }
                    }

                    if (split_exponents_.size() != (size_t)max_split_)
                    {
                        split_exponents_ = Vector<int>((size_t)max_split_, this->get_allocator(), this->get_gpuallocator());
                    }
                    splitted_ = true;
                }
                else if constexpr (std::is_same<split_type, float>::value)
                {
                    max_split_ = maxsplit;
                    if (split_matrices_fp32_.size() != (size_t)max_split_)
                    {
                        split_matrices_fp32_ = std::vector<Matrix<float>>(max_split_, Matrix<float>(this->shape(), cudart, this->get_gpuallocator()));
                        for (size_t i = 0; i < max_split_; ++i)
                        {
                            split_matrices_fp32_[i].allocateGPU(cudart);
                        }
                    }

                    if (split_exponents_.size() != (size_t)max_split_)
                    {
                        split_exponents_ = Vector<int>((size_t)max_split_, this->get_allocator(), this->get_gpuallocator());
                    }
                    splitted_ = true;
                }

            }
            
            ~MixedPrecisionMatrix() {};
        };
    } // namespace gpu
} // namespace lahva