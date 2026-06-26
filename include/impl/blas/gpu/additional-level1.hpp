/// @file additional-level1.hpp
/// @brief GPU-accelerated additional Level-1 operations beyond standard BLAS.
///
/// Provides GPU kernel declarations for supplementary vector-level linear algebra
/// utilities operating on GPU tensor objects. These extend standard BLAS Level-1 functionality.
/// Each function is provided for double and float precision with GPU runtime management.

#ifndef LAHVA_ADD_LEVEL1_HPP
#define LAHVA_ADD_LEVEL1_HPP
#include "linalg.hpp"
#include "../../../../src/gpu/gpu-utils/utils.hpp"
#include "kernels.cuh"
#include <vector>

namespace lahva
{
    namespace gpu
    {
        /// @brief Computes element-wise product (Hadamard product) of GPU tensor with itself in-place.
        ///
        /// Performs vecinout = vecinout ⊙ vecin (element-wise Hadamard multiplication).
        ///
        /// @tparam T Numerical type of tensors (double, float).
        /// @param cudart CUDA runtime instance
        /// @param vecin First input GPU tensor for element-wise product.
        /// @param vecinout Input and output GPU tensor (result stored here).
        template <typename T>
        void HadamardProduct(const CudaRuntime &cudart, const GPUTensor_<T> &vecin, GPUTensor_<T> &vecinout);

        /// @brief Computes element-wise product (Hadamard product) of two GPU tensors.
        ///
        /// Performs vecout = vecin ⊙ vecin2 (element-wise Hadamard multiplication).
        ///
        /// @tparam T Numerical type of tensors (double, float).
        /// @param cudart CUDA runtime instance
        /// @param vecin First input GPU tensor for element-wise product.
        /// @param vecin2 Second input GPU tensor for element-wise product.
        /// @param vecout Output GPU tensor to store the result.
        template <typename T>
        void HadamardProduct(const CudaRuntime &cudart, const GPUTensor_<T> &vecin, const GPUTensor_<T> &vecin2, GPUTensor_<T> &vecout);

        /// @brief GPU kernel for computing trace from diagonal elements.
        ///
        /// Accumulates diagonal elements into result vector using reduction kernel: trace(A) = sum(A_ii).
        ///
        /// @tparam T Numerical type (double, float).
        /// @param cudart CUDA runtime instance
        /// @param ndim Number of diagonal elements.
        /// @param diag Pointer to diagonal elements on GPU.
        /// @param vec Output vector on GPU for partial sums.
        template <typename T>
        void TraceKernelDiag(const CudaRuntime &cudart, unsigned long long ndim, const T *diag, T *vec);

        /// @brief Compute trace of a GPU tensor (trace(A) = sum(A_ii)).
        ///
        /// Computes the sum of all elements in the tensor using parallel reduction on GPU.
        /// Allocates a temporary reduction vector internally.
        ///
        /// @tparam U Host memory allocator type.
        /// @tparam V Device memory allocator type.
        /// @param cudart CUDA runtime instance
        /// @param diag Input GPU tensor containing elements to sum.
        /// @return Trace as a double-precision scalar.
        template <typename U, typename V>
        double ComputeTrace(const CudaRuntime &cudart, const GPUTensor<double, U, V> &diag)
        {
            int gridS = cudart.gridSize(diag.size(), 1);
            Vector<double, U, V> v(gridS);
            v.copy2device(cudart);
            TraceKernelDiag(cudart, diag.size(), diag.gpu_data(), v.gpu_data());
            v.copy2host(cudart);
            cudart.synchronize();
            return (double)v.sum();
        };

        /// @brief Compute trace of a GPU tensor using provided reduction vector (trace(A) = sum(A_ii)).
        ///
        /// Computes the sum of all elements in the tensor using parallel reduction on GPU.
        /// Uses the provided reduction vector for intermediate partial sums.
        ///
        /// @tparam U Host memory allocator type.
        /// @tparam V Device memory allocator type.
        /// @param cudart CUDA runtime instance
        /// @param diag Input GPU tensor containing elements to sum.
        /// @param v Pre-allocated reduction vector for partial sums (reused for efficiency).
        /// @return Trace as a double-precision scalar.
        template <typename U, typename V>
        double ComputeTrace(const CudaRuntime &cudart, const GPUTensor<double, U, V> &diag, Vector<double, U, V> &v)
        {
            v.copy2device(cudart);
            TraceKernelDiag(cudart, diag.size(), diag.gpu_data(), v.gpu_data());
            v.copy2host(cudart);
            cudart.synchronize();
            return (double)v.sum();
        };

        /// @brief Symmetrize a GPU matrix by computing (M + M^T) / 2 in-place.
        /// @tparam T Numerical type of matrix elements.
        /// @param cudart CUDA runtime instance
        /// @param Matrix GPU matrix to symmetrize in-place.
        template <typename T>
        void SymmetrizeMatrix(const CudaRuntime &cudart, Matrix_<T> &);

        /// @brief GPU kernel for computing Frobenius norm of a single matrix.
        ///
        /// Accumulates squared elements into result vector using reduction kernel: ||mat||_F = sqrt(sum(mat_ij^2)).
        ///
        /// @tparam T Numerical type (double, float).
        /// @param cudart CUDA runtime instance
        /// @param ndim Total number of elements in matrix.
        /// @param mat Pointer to matrix elements on GPU.
        /// @param vec Output vector on GPU for partial sums.
        template <typename T>
        void FrobeniusKernel(const CudaRuntime &cudart, const unsigned long long ndim, const T *mat, T *vec);

        /// @brief GPU kernel for computing difference norm between two matrices.
        ///
        /// Accumulates squared differences into result vector: ||mat1 - mat2||_F = sqrt(sum((mat1_ij - mat2_ij)^2)).
        ///
        /// @tparam T Numerical type (double, float).
        /// @param cudart CUDA runtime instance
        /// @param ndim Total number of elements in matrices.
        /// @param mat1 Pointer to first matrix elements on GPU.
        /// @param mat2 Pointer to second matrix elements on GPU.
        /// @param vec Output vector on GPU for partial sums.
        template <typename T>
        void FrobeniusKernel2(const CudaRuntime &cudart, const unsigned long long ndim, const T *mat1, const T *mat2, T *vec);

        /// @brief Compute Frobenius norm of a GPU tensor (||mat||_F = sqrt(sum(mat_ij^2))).
        ///
        /// Computes the Frobenius norm using parallel reduction on the GPU.
        /// Allocates a temporary reduction vector internally.
        ///
        /// @tparam T Numerical type (double, float).
        /// @tparam U Host memory allocator type.
        /// @tparam V Device memory allocator type.
        /// @param cudart CUDA runtime instance
        /// @param mat Input GPU tensor to compute norm of.
        /// @return The Frobenius norm as a scalar of type T.
        template <typename T, typename U, typename V>
        T FrobeniusNorm(const CudaRuntime &cudart, const GPUTensor<T, U, V> &mat)
        {
            check_device_alloc(cudart, mat);
            Vector<T, U, V> vec(cudart.gridSize(mat.size(), 1));
            ScaleVector(cudart, 0.0, vec);
            FrobeniusKernel<T>(cudart, mat.size(), mat.gpu_data(), vec.gpu_data());
            vec.copy2host(cudart);
            cudart.synchronize();
            T norm = vec.sum();
            return std::sqrt(norm);
        };

        /// @brief Compute Frobenius norm of difference between two GPU tensors (||mat - mat2||_F = sqrt(sum((mat_ij - mat2_ij)^2))).
        ///
        /// Computes the Frobenius norm of the difference using parallel reduction on the GPU.
        /// Allocates a temporary reduction vector internally.
        ///
        /// @tparam T Numerical type (double, float).
        /// @tparam U Host memory allocator type.
        /// @tparam V Device memory allocator type.
        /// @param cudart CUDA runtime instance
        /// @param mat First input GPU tensor.
        /// @param mat2 Second input GPU tensor to compare against.
        /// @return The Frobenius norm of the difference as a scalar of type T.
        template <typename T, typename U, typename V>
        T FrobeniusNorm(const CudaRuntime &cudart, const GPUTensor<T, U, V> &mat, const GPUTensor<T, U, V> &mat2)
        {
            // check_size_mm(mat, mat2);
            check_device_alloc(cudart, mat);
            check_device_alloc(cudart, mat2);
            Vector<T, U, V> vec(cudart.gridSize(mat.size(), 1));
            ScaleVector(cudart, 0.0, vec);
            FrobeniusKernel2<T>(cudart, mat.size(), mat.gpu_data(), mat2.gpu_data(), vec.gpu_data());
            vec.copy2host(cudart);
            cudart.synchronize();
            T norm = vec.sum();
            return std::sqrt(norm);
        };

        /// @brief Compute Frobenius norm of difference between two GPU tensors using provided reduction vector.
        ///
        /// Computes the Frobenius norm of the difference using parallel reduction on the GPU.
        /// Uses the provided reduction vector for intermediate partial sums.
        ///
        /// @tparam T Numerical type (double, float).
        /// @tparam U Host memory allocator type.
        /// @tparam V Device memory allocator type.
        /// @param cudart CUDA runtime instance
        /// @param mat First input GPU tensor.
        /// @param mat2 Second input GPU tensor to compare against.
        /// @param vec Pre-allocated reduction vector for partial sums (reused for efficiency).
        /// @return The Frobenius norm of the difference as a scalar of type T.
        template <typename T, typename U, typename V>
        T FrobeniusNorm(const CudaRuntime &cudart, const GPUTensor<T, U, V> &mat, const GPUTensor<T, U, V> &mat2, Vector<T, U, V> &vec)
        {
            // check_size_mm(mat, mat2);
            check_device_alloc(cudart, mat);
            check_device_alloc(cudart, mat2);
            if (vec.size() != cudart.gridSize(mat.size(), 1))
                vec = Vector<T, U, V>(cudart.gridSize(mat.size(), 1));
            ScaleVector(cudart, 0.0, vec);
            FrobeniusKernel2<T>(cudart, mat.size(), mat.gpu_data(), mat2.gpu_data(), vec.gpu_data());
            vec.copy2host(cudart);
            cudart.synchronize();
            T norm = vec.sum();
            return std::sqrt(norm);
        };

        /// @brief Compute Frobenius norm of a GPU tensor using provided reduction vector (||mat||_F = sqrt(sum(mat_ij^2))).
        ///
        /// Computes the Frobenius norm using parallel reduction on the GPU.
        /// Uses the provided reduction vector for intermediate partial sums.
        ///
        /// @tparam T Numerical type (double, float).
        /// @tparam U Host memory allocator type.
        /// @tparam V Device memory allocator type.
        /// @param cudart CUDA runtime instance
        /// @param mat Input GPU tensor to compute norm of.
        /// @param vec Pre-allocated reduction vector for partial sums (reused for efficiency).
        /// @return The Frobenius norm as a scalar of type T.
        template <typename T, typename U, typename V>
        T FrobeniusNorm(const CudaRuntime &cudart, const GPUTensor<T, U, V> &mat, Vector<T, U, V> &vec)
        {
            check_device_alloc(cudart, mat);
            ScaleVector(cudart, 0.0, vec);
            FrobeniusKernel(cudart, mat.size(), mat.gpu_data(), vec.gpu_data());
            vec.copy2host(cudart);
            cudart.synchronize();
            T norm = vec.sum();
            return std::sqrt(norm);
        };

        /// @brief Extract diagonal elements from a GPU matrix into a vector.
        /// Copies diagonal elements from matrix to vector.
        /// @tparam T Numerical type of matrix and vector elements.
        /// @param cudart CUDA runtime instance
        /// @param mat Input GPU matrix to extract diagonal from.
        /// @param vec Output GPU tensor to store diagonal elements.
        template <typename T>
        void GetDiagonal(const CudaRuntime &cudart, const Matrix_<T> &mat, GPUTensor_<T> &vec);

        /// @brief Set diagonal elements of a GPU matrix from a vector.
        /// Copies vector elements to the diagonal of the matrix.
        /// @tparam T Numerical type of matrix and vector elements.
        /// @param cudart CUDA runtime instance
        /// @param vec Input GPU tensor containing diagonal values.
        /// @param m Output GPU matrix with diagonal set (off-diagonal elements unchanged).
        template <typename T>
        void SetDiagonal(const CudaRuntime &cudart, const GPUTensor_<T> &vec, Matrix_<T> &m);

        /// @brief GPU kernel for computing trace of a matrix.
        ///
        /// Accumulates matrix elements along diagonal into result vector: trace(A) = sum(A_ii).
        ///
        /// @tparam T Numerical type of matrix and vector elements.
        /// @param cudart CUDA runtime instance
        /// @param ndim Matrix dimension (square matrix: n x n).
        /// @param diag Pointer to matrix elements on GPU.
        /// @param vec Output vector on GPU for partial sums.
        template <typename T>
        void TraceKernel(const CudaRuntime &cudart, unsigned long long ndim, const T *diag, T *vec);

        /// @brief Compute trace of a GPU matrix (trace(A) = sum(A_ii)).
        ///
        /// Computes the sum of diagonal elements using parallel reduction on GPU.
        /// Allocates a temporary reduction vector internally.
        ///
        /// @tparam T Numerical type of matrix and vector elements.
        /// @tparam U Host memory allocator type.
        /// @tparam V Device memory allocator type.
        /// @param cudart CUDA runtime instance
        /// @param m Input GPU matrix to compute trace of.
        /// @param use_diag If true, extract diagonal first; if false, compute directly from full matrix (default: false).
        /// @return The trace (sum of diagonal elements) as a double-precision scalar.
        template <typename T, typename U, typename V>
        double ComputeTrace(const CudaRuntime &cudart, const Matrix<T, U, V> &m, bool use_diag = false)
        {
            // Number of blocks in grid;
            int gridS = cudart.gridSize(m.shape().first, 1);
            Vector<T, U, V> v(gridS);
            v.copy2device(cudart);
            if (use_diag)
            {
                Vector<T, U, V> diag(m.shape().first, cudart);
                GetDiagonal(cudart, m, diag);
                TraceKernelDiag(cudart, diag.size(), diag.gpu_data(), v.gpu_data());
            }
            else
            {
                check_device_alloc(cudart, m);
                TraceKernel(cudart, m.shape().first, m.gpu_data(), v.gpu_data());
            }
            v.copy2host(cudart);
            cudart.synchronize();

            return (double)v.sum();
        }

        /// @brief Compute trace of a GPU matrix using provided reduction vector (trace(A) = sum(A_ii)).
        ///
        /// Computes the sum of diagonal elements using parallel reduction on GPU.
        /// Uses the provided reduction vector for intermediate partial sums.
        ///
        /// @tparam T Numerical type of matrix and vector elements.
        /// @tparam U Host memory allocator type.
        /// @tparam V Device memory allocator type.
        /// @param cudart CUDA runtime instance
        /// @param m Input GPU matrix to compute trace of.
        /// @param v Pre-allocated reduction vector for partial sums (reused for efficiency).
        /// @param use_diag If true, extract diagonal first; if false, compute directly from full matrix (default: false).
        /// @return The trace (sum of diagonal elements) as a double-precision scalar.
        template <typename T, typename U, typename V>
        double ComputeTrace(const CudaRuntime &cudart, const Matrix<T, U, V> &m, Vector<T, U, V> &v, bool use_diag = false)
        {
            v.copy2device(cudart);
            if (use_diag)
            {
                Vector<T, U, V> diag(m.shape().first, cudart);
                GetDiagonal(cudart, m, diag);
                TraceKernelDiag(cudart, diag.size(), diag.gpu_data(), v.gpu_data());
            }
            else
            {
                check_device_alloc(cudart, m);
                TraceKernel(cudart, m.shape().first, m.gpu_data(), v.gpu_data());
            }
            v.copy2host(cudart);
            cudart.synchronize();

            return (double)v.sum();
        }

        /// @brief Decompose vector into two lower-precision components using Ozaki algorithm.
        ///
        /// Splits a high-precision vector into two lower-precision mantissa components with
        /// corresponding exponent coefficients for accurate mixed-precision computation.
        /// Uses the Ozaki-Ahrens decomposition algorithm: splits based on the maximum
        /// element's exponent and precision characteristics.
        ///
        /// @tparam inprec Input high-precision type
        /// @tparam outprec Output lower-precision type
        /// @param cudart CUDA runtime instance
        /// @param in Input GPU tensor in high precision to decompose.
        /// @param out1 First output component vector in lower precision (mantissa part 1).
        /// @param out2 Second output component vector in lower precision (mantissa part 2).
        /// @param coeff Output vector for exponent coefficients (one per decomposition level).
        ///
        /// @note The decomposition satisfies: in ≈ sigma_1 * out1 + sigma_2 * out2
        ///       where sigma_i = 2^coeff[i]
        /// @note Both output vectors are allocated on GPU and overwritten with decomposed values.
        /// @pre All input tensors must be allocated on GPU device.
        template <typename inprec, typename outprec>
        void DecomposeVector2MP(const CudaRuntime &cudart, const GPUTensor_<inprec> &in, GPUTensor_<outprec> &out1, GPUTensor_<outprec> &out2, GPUTensor_<int>& coeff);

        /// @brief Decompose matrix into multiple lower-precision components using Ozaki algorithm.
        ///
        /// Splits a high-precision matrix iteratively into multiple lower-precision component matrices
        /// for accurate mixed-precision computation via the Ozaki-Ahrens decomposition algorithm.
        ///
        /// @tparam inprec Input high-precision type
        /// @tparam outprec Output lower-precision type
        /// @tparam Allocator Host memory allocator type for output matrices.
        /// @tparam GPUAllocator Device memory allocator type for output matrices.
        /// @param cudart CUDA runtime instance
        /// @param in Input high-precision GPU matrix to decompose (modified in-place as residual).
        /// @param out Vector of output matrices in lower precision (one per decomposition level).
        /// @param coeff Output vector for exponent coefficients (one per decomposition level).
        /// @param maxsplit Maximum number of decomposition levels to compute.
        ///
        /// @note The decomposition satisfies: in ≈ sum_i (sigma_i * out[i])
        ///       where sigma_i = 2^coeff[i] * 0.75
        /// @note The input matrix `in` is overwritten with the residual at each level.
        /// @pre All matrices must be allocated on GPU device.
        template <typename inprec, typename outprec, typename Allocator, typename GPUAllocator>
        void SplitMatrix(const CudaRuntime &cudart, Matrix_<inprec> &in, std::vector<Matrix<outprec, Allocator, GPUAllocator>> &out1, GPUTensor_<int> &split_exponents, int max_split);

        /// @brief Decompose tensor into two lower-precision component vectors with auto-allocation.
        ///
        /// Convenience wrapper around DecomposeVector2MP that automatically allocates or resizes
        /// output vectors to match the input tensor size. Useful when output vector sizes are unknown.
        ///
        /// @tparam T Input precision type.
        /// @tparam U Host memory allocator type.
        /// @tparam V Device memory allocator type.
        /// @tparam Tout Output precision type for components.
        /// @param cudart CUDA runtime instance
        /// @param min Input GPU tensor to decompose (any dimensionality, treated as 1D).
        /// @param mout1 First output component vector (auto-allocated/resized if needed).
        /// @param mout2 Second output component vector (auto-allocated/resized if needed).
        template <typename T, typename U, typename V, typename Tout>
        void DecomposeMatrix(const CudaRuntime &cudart, const GPUTensor<T, U, V> &min, Vector<Tout, U, V> &mout1, Vector<Tout, U, V> &mout2)
        {
            if (min.size() != mout1.size())
                Vector<Tout, U, V> mout1(min.size(), cudart);

            if (min.size() != mout2.size())
                Vector<Tout, U, V> mout2(min.size(), cudart);

            DecomposeVector2MP<T, Tout>(cudart, min, mout1, mout2);
        }

        /// @brief Apply custom operation element-wise to GPU tensor in-place.
        /// Applies user-defined operation to each element of tensor.
        /// @tparam T Numerical type of tensor elements.
        /// @tparam op Functor type for operation to apply.
        /// @param cudart CUDA runtime instance
        /// @param in GPU tensor to modify in-place.
        /// @param operation Functor instance defining operation (default: op()).
        template <typename T, class op>
        void ApplyKernel(const CudaRuntime &cudart, GPUTensor_<T> &in, op operation = op());

        /// @brief Compute sum of all GPU tensor elements using parallel reduction.
        /// Accumulates all elements into a single scalar via GPU reduction kernel.
        /// @tparam T Numerical type of tensor elements.
        /// @param cudart CUDA runtime instance
        /// @param in Input GPU tensor to sum.
        /// @param res Temporary reduction vector for partial sums.
        /// @return Sum of all tensor elements.
        template <typename T>
        T Sum_(const CudaRuntime &cudart, const GPUTensor_<T> &in, GPUTensor_<T> &res);

        /// @brief Find maximum element in GPU tensor using parallel reduction.
        /// Locates and returns the largest element in tensor via GPU reduction.
        /// @tparam T Numerical type of tensor elements.
        /// @param cudart CUDA runtime instance
        /// @param in Input GPU tensor to search.
        /// @param res Temporary reduction vector for partial maxima.
        /// @return Maximum element value.
        template <typename T>
        T MaxElement_(const CudaRuntime &cudart, const GPUTensor_<T> &in, GPUTensor_<T> &res);

        /// @brief Find minimum element in GPU tensor using parallel reduction.
        /// Locates and returns the smallest element in tensor via GPU reduction.
        /// @tparam T Numerical type of tensor elements.
        /// @param cudart CUDA runtime instance
        /// @param in Input GPU tensor to search.
        /// @param res Temporary reduction vector for partial minima.
        /// @return Minimum element value.
        template <typename T>
        T MinElement_(const CudaRuntime &cudart, const GPUTensor_<T> &in, GPUTensor_<T> &res);

        /// @brief Compute sum of all GPU tensor elements (convenience wrapper with auto-allocation).
        ///
        /// Allocates temporary reduction vector using mapped host memory automatically.
        ///
        /// @tparam T Numerical type of tensor elements.
        /// @tparam U Host memory allocator type.
        /// @tparam V Device memory allocator type.
        /// @param cudart CUDA runtime instance
        /// @param in Input GPU tensor to sum.
        /// @return Sum of all tensor elements.
        template <typename T, typename U, typename V>
        T Sum(const CudaRuntime &cudart, const GPUTensor<T, U, V> &in)
        {
            unsigned long long n = in.size();
            unsigned long long blockSize = cudart.blockSize();
            size_t blocksPerGrid = std::ceil((1. * n) / blockSize);
            Vector<T, CudaHostAllocatorMapped<T>> v(blocksPerGrid);
            return Sum_(cudart, in, v);
        };

        /// @brief Compute sum of all GPU tensor elements using provided reduction vector.
        ///
        /// Computes sum with caller-provided reduction vector for intermediate partial sums.
        ///
        /// @tparam T Numerical type of tensor elements.
        /// @tparam U Host memory allocator type.
        /// @tparam V Device memory allocator type.
        /// @param cudart CUDA runtime instance
        /// @param in Input GPU tensor to sum.
        /// @param v Pre-allocated reduction vector for partial sums.
        /// @return Sum of all tensor elements.
        template <typename T, typename U, typename V>
        T Sum(const CudaRuntime &cudart, const GPUTensor<T, U, V> &in, Vector<T, U, V> &v)
        {
            unsigned long long n = in.size();
            unsigned long long blockSize = cudart.blockSize();
            size_t blocksPerGrid = std::ceil((1. * n) / blockSize);
            assert(v.size() >= blocksPerGrid);
            return Sum_(cudart, in, v);
        };

        /// @brief Find maximum element in GPU tensor using provided reduction vector.
        ///
        /// Locates and returns the largest element in tensor via GPU reduction.
        /// Automatically resizes provided reduction vector if needed.
        ///
        /// @tparam T Numerical type of tensor elements.
        /// @tparam U Host memory allocator type.
        /// @tparam V Device memory allocator type.
        /// @param cudart CUDA runtime instance
        /// @param in Input GPU tensor to search.
        /// @param v Pre-allocated reduction vector for partial maxima (resized if necessary).
        /// @return Maximum element value.
        template <typename T, typename U, typename V>
        T MaxElement(const CudaRuntime &cudart, const GPUTensor<T, U, V> &in, Vector<T, U, V> &v)
        {
            unsigned long long n = in.size();
            unsigned long long blockSize = cudart.blockSize();
            size_t blocksPerGrid = std::ceil((1. * n) / blockSize);
            if (v.size() != blocksPerGrid)
                v = Vector<T, U, V>(blocksPerGrid);
            T res = MaxElement_(cudart, in, v);
            cudart.setblockSize(512);
            return res;
        };

        /// @brief Find minimum element in GPU tensor using provided reduction vector.
        ///
        /// Locates and returns the smallest element in tensor via GPU reduction.
        /// Automatically resizes provided reduction vector if needed.
        ///
        /// @tparam T Numerical type of tensor elements.
        /// @tparam U Host memory allocator type.
        /// @tparam V Device memory allocator type.
        /// @param cudart CUDA runtime instance
        /// @param in Input GPU tensor to search.
        /// @param v Pre-allocated reduction vector for partial minima (resized if necessary).
        /// @return Minimum element value.
        template <typename T, typename U, typename V>
        T MinElement(const CudaRuntime &cudart, const GPUTensor<T, U, V> &in, Vector<T, U, V> &v)
        {
            unsigned long long n = in.size();
            unsigned long long blockSize = cudart.blockSize();
            size_t blocksPerGrid = std::ceil((1. * n) / blockSize);
            if (v.size() != blocksPerGrid)
                v = Vector<T, U, V>(blocksPerGrid);
            return MinElement_(cudart, in, v);
        };

        /// @brief Find maximum element in GPU tensor using parallel reduction (convenience wrapper).
        ///
        /// Allocates temporary reduction vector internally.
        ///
        /// @tparam T Numerical type of tensor elements.
        /// @tparam U Host memory allocator type.
        /// @tparam V Device memory allocator type.
        /// @param cudart CUDA runtime instance
        /// @param in Input GPU tensor to search.
        /// @return Maximum element value.
        template <typename T, typename U, typename V>
        T MaxElement(const CudaRuntime &cudart, const GPUTensor<T, U, V> &in)
        {
            unsigned long long n = in.size();
            unsigned long long blockSize = cudart.blockSize();
            size_t blocksPerGrid = std::ceil((1. * n) / blockSize);
            Vector<T, U, V> v(blocksPerGrid);
            T res = MaxElement_(cudart, in, v);
            cudart.setblockSize(512);
            return res;
        };

        /// @brief Find minimum element in GPU tensor using parallel reduction (convenience wrapper).
        ///
        /// Allocates temporary reduction vector internally.
        ///
        /// @tparam T Numerical type of tensor elements.
        /// @tparam U Host memory allocator type.
        /// @tparam V Device memory allocator type.
        /// @param cudart CUDA runtime instance
        /// @param in Input GPU tensor to search.
        /// @return Minimum element value.
        template <typename T, typename U, typename V>
        T MinElement(const CudaRuntime &cudart, const GPUTensor<T, U, V> &in)
        {
            unsigned long long n = in.size();
            unsigned long long blockSize = cudart.blockSize();
            size_t blocksPerGrid = std::ceil((1. * n) / blockSize);
            Vector<T, U, V> v(blocksPerGrid);
            return MinElement_(cudart, in, v);
        };

        /// @brief Merge Ozaki decomposed components into final result (GPU).
        /// Reconstructs high-precision result from multiple lower-precision components.
        /// Used in Ozaki-Ahrens iterative refinement algorithm.
        /// @param cudart CUDA runtime instance
        /// @param ndim Vector dimension to process.
        /// @param nsplit Number of component vectors to merge.
        /// @param alphas Exponent scaling factors on GPU (device pointer).
        /// @param as Array of component vectors on GPU (array of device pointers).
        /// @param b Output result vector on GPU (device pointer).
        void MergeOzaki(const CudaRuntime &cudart, unsigned long long ndim, unsigned int nsplit, const double *alphas, const float **as, double *b);
    } // namespace gpu

} // namespace lahva
#endif