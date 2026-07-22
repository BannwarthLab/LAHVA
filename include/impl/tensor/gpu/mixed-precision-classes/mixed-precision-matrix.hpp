/// @file mixed-precision-matrix.hpp
/// @brief GPU-based mixed-precision matrix tensor implementation.
///
/// Provides the MixedPrecisionMatrix class for matrices with dual precision representation.
/// Supports computation in reduced precision (e.g., float32) while maintaining higher precision
/// (e.g., float64) for accuracy. Optimizes performance for mixed-precision linear algebra algorithms.

#pragma once

#include <vector>
#include "impl/tensor/allocators.hpp"
#include "impl/tensor/cpu/matrix.hpp"
#include "impl/tensor/cpu/tensor.hpp"
#include "impl/tensor/gpu/gputensor.hpp"
#include "impl/tensor/gpu/matrix.hpp"

namespace lahva
{
    namespace gpu
    {

        template <typename high_prec, class Allocator = CudaHostAllocator<high_prec>, class GPUAllocator = CudaDeviceAsyncAllocator<high_prec>>
        class MixedPrecisionMatrix : public lahva::gpu::Matrix<high_prec, Allocator, GPUAllocator>
        {
        protected:
            using Matrix<high_prec, Allocator, GPUAllocator>::n_rows_;
            using Matrix<high_prec, Allocator, GPUAllocator>::n_cols_;
            using Matrix<high_prec, Allocator, GPUAllocator>::data_id_;
            using Matrix<high_prec, Allocator, GPUAllocator>::data_size_;
            using Matrix<high_prec, Allocator, GPUAllocator>::check_size_;

        public:
            mutable size_t max_split_ = 0; // maximum number of splits for the mixed precision matrix
            mutable std::vector<Matrix<__half>> split_matrices_fp16_; // stores the low precision matrices
            mutable std::vector<Matrix<float>> split_matrices_fp32_;
            mutable bool splitted_fp16_ = false; // indicates whether the matrix has been split
            mutable bool splitted_fp32_ = false;       // indicates whether the matrix has been split
            mutable Vector<int> split_exponents_; // stores the indices for splitting the matrix
            size_t mod_value_ = 8;               // value to which the rows and columns are rounded
        public:
            using Matrix<high_prec, Allocator, GPUAllocator>::shape;

            MixedPrecisionMatrix() : Matrix<high_prec, Allocator, GPUAllocator>{} {};
            MixedPrecisionMatrix(const Shape &shape, const Allocator &alloc = Allocator(), const GPUAllocator &gpualloc = GPUAllocator());
            MixedPrecisionMatrix(const Shape &shape, const CudaRuntime &cudart, const GPUAllocator &gpualloc = GPUAllocator())
                : Matrix<high_prec, Allocator, GPUAllocator>(shape, cudart, gpualloc)
            {
            };
            MixedPrecisionMatrix(const Matrix<high_prec, Allocator, GPUAllocator> &other) : 
            Matrix<high_prec, Allocator, GPUAllocator>(other)
            {
            }
            MixedPrecisionMatrix(const Shape &shape, std::initializer_list<high_prec> init, bool row_major = false, const Allocator &alloc = Allocator(), const GPUAllocator &gpualloc = GPUAllocator())
                : Matrix<high_prec, Allocator, GPUAllocator>(shape, init, row_major, alloc, gpualloc)
            {
            }

            MixedPrecisionMatrix(const Shape &shape, high_prec *data_ptr, bool take_ownership = false, const Allocator &alloc = Allocator(), const GPUAllocator &gpualloc = GPUAllocator())
                : Matrix<high_prec, Allocator, GPUAllocator>(shape, data_ptr, take_ownership, alloc, gpualloc)
            {
            };

            MixedPrecisionMatrix(const MixedPrecisionMatrix &other) : 
            Matrix<high_prec, Allocator, GPUAllocator>(other),
            max_split_{other.max_split_},
            split_matrices_fp16_{other.split_matrices_fp16_},
            split_matrices_fp32_{other.split_matrices_fp32_},
            splitted_fp16_{other.splitted_fp16_},
            splitted_fp32_{other.splitted_fp32_},
            split_exponents_{other.split_exponents_},
            mod_value_{other.mod_value_}
            {

            }

            MixedPrecisionMatrix(MixedPrecisionMatrix &&other) : 
            Matrix<high_prec, Allocator, GPUAllocator>(std::move(other)),
            max_split_{other.max_split_},
            split_matrices_fp16_{std::move(other.split_matrices_fp16_)},
            split_matrices_fp32_{std::move(other.split_matrices_fp32_)},
            splitted_fp16_{other.splitted_fp16_},
            splitted_fp32_{other.splitted_fp32_},
            split_exponents_{std::move(other.split_exponents_)},
            mod_value_{other.mod_value_}
            {
                other.splitted_fp16_ = false;
                other.splitted_fp32_ = false;
                other.max_split_ = 0;
            }

            MixedPrecisionMatrix &operator=(const MixedPrecisionMatrix &other)
            {
                if (this != &other)
                {
                    Matrix<high_prec, Allocator, GPUAllocator>::operator=(other);
                    max_split_ = other.max_split_;
                    split_matrices_fp16_ = other.split_matrices_fp16_;
                    split_matrices_fp32_ = other.split_matrices_fp32_;
                    splitted_fp16_ = other.splitted_fp16_;
                    splitted_fp32_ = other.splitted_fp32_;
                    split_exponents_ = other.split_exponents_;
                    mod_value_ = other.mod_value_;
                }
                return *this;
            };

            MixedPrecisionMatrix &operator=(MixedPrecisionMatrix &&other)
            {
                if (this != &other)
                {
                    Matrix<high_prec, Allocator, GPUAllocator>::operator=(std::move(other));
                    max_split_ = other.max_split_;
                    split_matrices_fp16_ = std::move(other.split_matrices_fp16_);
                    split_matrices_fp32_ = std::move(other.split_matrices_fp32_);
                    splitted_fp16_ = other.splitted_fp16_;
                    splitted_fp32_ = other.splitted_fp32_;
                    split_exponents_ = std::move(other.split_exponents_);
                    mod_value_ = other.mod_value_;

                    other.splitted_fp16_ = false;
                    other.splitted_fp32_ = false;
                    other.max_split_ = 0;
                }
                return *this;
            };
            
            template <typename split_type>
            void split(const CudaRuntime &cudart, int max_split, Matrix_<high_prec> &buffer) const;

            void merge(const CudaRuntime &cudart, const high_prec *alphas, high_prec ini_beta);
            void merge(const CudaRuntime &cudart);
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
                
                if (i >= split_exponents_.size())
                {
                    throw std::out_of_range("Index out of range for split exponents.");
                }
                return split_exponents_[i];
            }            

            template <typename split_type>
            const Matrix<split_type> &getSplitMatrix(size_t i) const
            {
                if (!splitted_fp16_ && !splitted_fp32_)
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
            void resetSplit() const
            {
                splitted_fp16_ = false;
                
            }

            template <typename split_type>
            void createSplitMatrices(const CudaRuntime &cudart, int maxsplit) const
            {
                Shape s = this->shape();
                
                if constexpr (std::is_same<split_type, __half>::value)
                {
                    max_split_ = maxsplit;
                    if (split_matrices_fp16_.size() != (size_t)max_split_)
                    {
                        split_matrices_fp16_ = std::vector<Matrix<__half>>(max_split_, Matrix<__half>(s, cudart, this->get_gpuallocator()));
                        for (size_t i = 0; i < max_split_; ++i)
                        {
                            split_matrices_fp16_[i].allocateGPU(cudart);
                        }
                    }
                    if (split_exponents_.size() < (size_t)max_split_)
                    {
                        split_exponents_ = Vector<int>((size_t)max_split_, this->get_allocator(), this->get_gpuallocator());
                    }
                    splitted_fp16_  = true;
                }
                else if constexpr (std::is_same<split_type, float>::value)
                {
                    max_split_ = maxsplit;
                    if (split_matrices_fp32_.size() != (size_t)max_split_)
                    {
                        split_matrices_fp32_ = std::vector<Matrix<float>>(max_split_, Matrix<float>(s, cudart, this->get_gpuallocator()));
                        for (size_t i = 0; i < max_split_; ++i)
                        {
                            split_matrices_fp32_[i].allocateGPU(cudart);
                        }
                    }

                    splitted_fp32_ = true;
                }

            }

            template <typename split_type>
            void deallocateSplitMatrices(const CudaRuntime &cudart) const
            {
                if constexpr (std::is_same<split_type, __half>::value)
                {
                    for (size_t i = 0; i < split_matrices_fp16_.size(); ++i)
                    {
                        split_matrices_fp16_[i].deallocateGPU(cudart);
                    }
                    splitted_fp16_ = false;
                }
                else if constexpr (std::is_same<split_type, float>::value)
                {
                    for (size_t i = 0; i < split_matrices_fp32_.size(); ++i)
                    {
                        split_matrices_fp32_[i].deallocateGPU(cudart);
                    }
                    splitted_fp32_ = false;
                }
            }
            
            ~MixedPrecisionMatrix() 
            {
            };
        };
    } // namespace gpu
}// namespace lahva