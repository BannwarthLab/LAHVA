/// @file mixed-precision-matrix.hpp
/// @brief GPU-based mixed-precision matrix tensor implementation.
///
/// Provides the MixedPrecisionMatrix class for matrices with dual precision representation.
/// Supports computation in reduced precision (e.g., float32) while maintaining higher precision
/// (e.g., float64) for accuracy. Optimizes performance for mixed-precision linear algebra algorithms.

#pragma once

#include "impl/tensor/allocators.hpp"
#include "impl/tensor/cpu/matrix.hpp"
#include "impl/tensor/gpu/matrix.hpp"

namespace lahva
{
    namespace gpu
    {

        /// @brief GPU-based mixed-precision matrix for heterogeneous precision computations
        ///
        /// Supports dual-precision representation where computations can be done in reduced
        /// precision (e.g., float32 or float16) while maintaining higher precision (e.g., float64)
        /// representation. Enables performance optimization for mixed-precision linear algebra algorithms.
        /// Can split the matrix into smaller precision blocks for refined computations.
        ///
        /// @tparam high_prec high precision data type (e.g., double, float)
        /// @tparam Allocator host (CPU) memory allocator type (default: CudaHostAllocator)
        /// @tparam GPUAllocator device (GPU) memory allocator type (default: CudaDeviceAsyncAllocator)
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
            /// @brief Maximum number of splits for mixed precision blocks
            mutable size_t max_split_ = 0;

            /// @brief Stores split matrices in float16 (half) precision
            mutable std::vector<Matrix<__half>> split_matrices_fp16_;

            /// @brief Stores split matrices in float32 precision
            mutable std::vector<Matrix<float>> split_matrices_fp32_;

            /// @brief Flag indicating if matrix has been split into float16 blocks
            mutable bool splitted_fp16_ = false;

            /// @brief Flag indicating if matrix has been split into float32 blocks
            mutable bool splitted_fp32_ = false;

            /// @brief Exponent values for split blocks (used for scaling)
            mutable Vector<int> split_exponents_;

            /// @brief Alignment value for rounding matrix dimensions in splits (default: 8)
            size_t mod_value_ = 8;

        public:
            using Matrix<high_prec, Allocator, GPUAllocator>::shape;

            /// @brief Default constructor for mixed-precision matrix
            MixedPrecisionMatrix() : Matrix<high_prec, Allocator, GPUAllocator>{} {};

            /// @brief Construct a mixed-precision matrix with specified shape
            /// @param[in] shape matrix dimensions (rows, cols)
            /// @param[in] alloc host (CPU) memory allocator
            /// @param[in] gpualloc device (GPU) memory allocator
            MixedPrecisionMatrix(const Shape &shape, const Allocator &alloc = Allocator(), const GPUAllocator &gpualloc = GPUAllocator());

            /// @brief Construct a mixed-precision matrix using CUDA runtime
            /// @param[in] shape matrix dimensions (rows, cols)
            /// @param[in] cudart CUDA runtime instance for GPU operations
            /// @param[in] gpualloc device (GPU) memory allocator
            MixedPrecisionMatrix(const Shape &shape, const CudaRuntime &cudart, const GPUAllocator &gpualloc = GPUAllocator())
                : Matrix<high_prec, Allocator, GPUAllocator>(shape, cudart, gpualloc)
            {
            };

            /// @brief Construct a mixed-precision matrix from a standard matrix
            /// @param[in] other source matrix to copy
            MixedPrecisionMatrix(const Matrix<high_prec, Allocator, GPUAllocator> &other) :
            Matrix<high_prec, Allocator, GPUAllocator>(other)
            {
            }

            /// @brief Construct a mixed-precision matrix from initializer list
            /// @param[in] shape matrix dimensions (rows, cols)
            /// @param[in] init initializer list with matrix values
            /// @param[in] row_major if true, interpret init as row-major; else column-major
            /// @param[in] alloc host (CPU) memory allocator
            /// @param[in] gpualloc device (GPU) memory allocator
            MixedPrecisionMatrix(const Shape &shape, std::initializer_list<high_prec> init, bool row_major = false, const Allocator &alloc = Allocator(), const GPUAllocator &gpualloc = GPUAllocator())
                : Matrix<high_prec, Allocator, GPUAllocator>(shape, init, row_major, alloc, gpualloc)
            {
            }

            /// @brief Construct a mixed-precision matrix from raw data pointer
            /// @param[in] shape matrix dimensions (rows, cols)
            /// @param[in] data_ptr pointer to matrix data
            /// @param[in] take_ownership if false, data is not freed when matrix is destructed
            /// @param[in] alloc host (CPU) memory allocator
            /// @param[in] gpualloc device (GPU) memory allocator
            MixedPrecisionMatrix(const Shape &shape, high_prec *data_ptr, bool take_ownership = false, const Allocator &alloc = Allocator(), const GPUAllocator &gpualloc = GPUAllocator())
                : Matrix<high_prec, Allocator, GPUAllocator>(shape, data_ptr, take_ownership, alloc, gpualloc)
            {
            };

            /// @brief Copy constructor for mixed-precision matrix
            /// @param[in] other source matrix to copy
            MixedPrecisionMatrix(const MixedPrecisionMatrix &other) :
            Matrix<high_prec, Allocator, GPUAllocator>(other),
            max_split_{other.max_split_},
            split_matrices_fp16_{other.split_matrices_fp16_},
            split_matrices_fp32_{other.split_matrices_fp32_},
            splitted_fp16_{other.splitted_fp16_},
            splitted_fp32_{other.splitted_fp32_},
            split_exponents_{other.split_exponents_},
            mod_value_{other.mod_value_},
            is_merged_{other.is_merged_},
            cudart_{other.cudart_}
            {

            }

            /// @brief Move constructor for mixed-precision matrix
            /// @param[in] other source matrix to move from
            MixedPrecisionMatrix(MixedPrecisionMatrix &&other) :
            Matrix<high_prec, Allocator, GPUAllocator>(std::move(other)),
            max_split_{other.max_split_},
            split_matrices_fp16_{std::move(other.split_matrices_fp16_)},
            split_matrices_fp32_{std::move(other.split_matrices_fp32_)},
            splitted_fp16_{other.splitted_fp16_},
            splitted_fp32_{other.splitted_fp32_},
            split_exponents_{std::move(other.split_exponents_)},
            mod_value_{other.mod_value_},
            is_merged_{other.is_merged_},
            cudart_{other.cudart_}
            {
                other.splitted_fp16_ = false;
                other.splitted_fp32_ = false;
                other.max_split_ = 0;
                other.is_merged_ = true;
                other.cudart_ = nullptr;
            }

            /// @brief Copy assignment operator
            /// @param[in] other source matrix
            /// @return reference to this matrix
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
                    is_merged_ = other.is_merged_;
                    cudart_ = other.cudart_;
                }
                return *this;
            };

            /// @brief Move assignment operator
            /// @param[in] other source matrix to move from
            /// @return reference to this matrix
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
                    is_merged_ = other.is_merged_;
                    cudart_ = other.cudart_;

                    other.splitted_fp16_ = false;
                    other.splitted_fp32_ = false;
                    other.max_split_ = 0;
                    other.is_merged_ = true;
                    other.cudart_ = nullptr;
                }
                return *this;
            };

            /// @brief Split matrix into smaller precision blocks
            /// @tparam split_type target precision type for splits (__half or float)
            /// @param[in] cudart CUDA runtime instance
            /// @param[in] max_split maximum number of blocks to create
            /// @param[in] buffer temporary buffer matrix for computation
            template <typename split_type>
            void split(const CudaRuntime &cudart, int max_split, Matrix_<high_prec> &buffer) const;

            /// @brief Merge split matrices back with weighted scaling
            /// @param[in] cudart CUDA runtime instance
            /// @param[in] alphas scaling factors for each split block
            /// @param[in] ini_beta initial beta value for merge operation
            void merge(const CudaRuntime &cudart, const high_prec *alphas, high_prec ini_beta);

            /// @brief Merge split matrices back to original precision
            /// @param[in] cudart CUDA runtime instance
            void merge(const CudaRuntime &cudart);

            /// @brief Get the number of split blocks of a given precision type
            /// @tparam split_type precision type to query (__half or float)
            /// @return number of split blocks of the specified type
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

            /// @brief Get the exponent value for a split block
            /// @param[in] i index of the split block
            /// @return exponent value used for scaling this split block
            /// @throws std::out_of_range if index is out of range
            int getSplitExponent(size_t i) const
            {

                if (i >= split_exponents_.size())
                {
                    throw std::out_of_range("Index out of range for split exponents.");
                }
                return split_exponents_[i];
            }

            /// @brief Get a specific split matrix block
            /// @tparam split_type precision type of split blocks (__half or float)
            /// @param[in] i index of the split block
            /// @return const reference to the split matrix at index i
            /// @throws std::runtime_error if matrix has not been split
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

            /// @brief Get all split matrices of a given precision type
            /// @tparam split_type precision type of split blocks (__half or float)
            /// @return reference to vector of all split matrices of specified type
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

            /// @brief Reset split state of the matrix
            void resetSplit() const
            {
                splitted_fp16_ = false;

            }

            /// @brief Implementation: Create and allocate split precision matrices on GPU
            /// Allocates GPU memory for specified number of split blocks
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

            /// @brief Implementation: Release GPU memory for split precision matrices
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

            /// @brief Ensure runtime is captured for lazy merging
            /// Stores the runtime pointer on first call; subsequent calls are no-ops
            void ensure_runtime(const CudaRuntime &cudart) {
                if (!cudart_ && !is_merged_) {
                    cudart_ = &cudart;
                }
            }

            /// @brief Mark the high-precision base matrix as stale (out of sync with split components)
            void mark_stale() {
                is_merged_ = false;
            }

            /// @brief Override data() to auto-merge if stale (matches base class signature)
            high_prec* data() const override {
                if (!is_merged_ && cudart_) {
                    const_cast<MixedPrecisionMatrix*>(this)->merge(*cudart_);
                    const_cast<MixedPrecisionMatrix*>(this)->is_merged_ = true;
                }
                return Matrix<high_prec, Allocator, GPUAllocator>::data();
            }

        private:
            /// @brief Tracks whether the high-precision base matrix is in sync with split components
            mutable bool is_merged_ = true;

            /// @brief Pointer to CudaRuntime for lazy merging (captured on first operation)
            const CudaRuntime* cudart_ = nullptr;

        public:
            /// @brief Destructor implementation - base class handles memory cleanup
            ~MixedPrecisionMatrix()
            {
            };
        };
    } // namespace gpu
}// namespace lahva