/// @file additional-level3.hpp
/// @brief GPU-accelerated additional Level-3 operations beyond standard BLAS.
///
/// Provides GPU kernel declarations for supplementary matrix-matrix operations
/// operating on GPU matrix objects. These extend standard BLAS Level-3 functionality.
/// Each function is provided for double and float precision with GPU runtime management.

#pragma once
#include "impl/blas/gpu/additional-level1.hpp"
#include "impl/blas/gpu/additional-level2.hpp"
#include "impl/blas/gpu/level1.hpp"
#include "impl/blas/gpu/level3.hpp"
#include "linalg.hpp"
#include "runtime.hpp"
#include "additional-level3-add.hpp"

namespace lahva
{
    namespace gpu
    {

        /// @brief Computes symmetric matrix product with diagonal scaling using mixed precision.
        ///
        /// Performs scaled multiplication D1 * M1 * D2 * M2 with symmetrization.
        /// Combines scaled matrices maintaining symmetry property.
        ///
        /// @tparam Allocator Host memory allocator type for vectors.
        /// @tparam GPUAllocator Device memory allocator type for vectors.
        /// @tparam All2 Host memory allocator type for matrices.
        /// @tparam GPUAll2 Device memory allocator type for matrices.
        /// @param cudart CUDA runtime instance
        /// @param d1 First diagonal scaling vector (Vector<double>).
        /// @param m1 First matrix in single-precision (Matrix<float>).
        /// @param d2 Second diagonal scaling vector (Vector<double>).
        /// @param m2 Second matrix in single-precision (Matrix<float>).
        /// @param dout Output diagonal result from Hadamard product (Vector<double>).
        /// @param mout Output matrix with scaled product (destination / input-output, Matrix<float>).
        /// @param fast Enable fast mode optimization (always true).
        template<typename Allocator, typename GPUAllocator, typename All2, typename GPUAll2>
        void MPSymMatrixMatrixMultiplication(const CudaRuntime& cudart, const Vector<double, Allocator, GPUAllocator>& d1, const Matrix<float, All2, GPUAll2>& m1,
                                             const Vector<double, Allocator, GPUAllocator>& d2, const Matrix<float, All2, GPUAll2>& m2,
                                             Vector<double, Allocator, GPUAllocator>& dout, Matrix<float, All2, GPUAll2>& mout, bool fast = true)
        {
            fast = true;
            MatrixMatrixProduct(cudart, m1, m2, mout);
            HadamardProduct(cudart, d1, d2, dout);
            SymmetrizedON2ScalingProductGPU(cudart, d1, m1, d2, m2, mout);
            SymmetrizeMatrix(cudart, mout);
            
        };

        /// @brief Computes symmetric matrix product with diagonal scaling using mixed-precision matrices.
        ///
        /// Performs scaled multiplication D1 * M1 * D2 * M2 with symmetrization using iterative refinement.
        ///
        /// @tparam Allocator Host memory allocator type for vectors.
        /// @tparam GPUAllocator Device memory allocator type for vectors.
        /// @tparam All2 Host memory allocator type for matrices.
        /// @tparam GPUAll2 Device memory allocator type for matrices.
        /// @param cudart CUDA runtime instance
        /// @param mp_rt Mixed-precision runtime configuration controlling precision level.
        /// @param d1 First diagonal scaling vector (Vector<double>).
        /// @param m1 First matrix with mixed-precision representation (MixedPrecisionMatrix<float>).
        /// @param d2 Second diagonal scaling vector (Vector<double>).
        /// @param m2 Second matrix with mixed-precision representation (MixedPrecisionMatrix<float>).
        /// @param dout Output diagonal result from Hadamard product (Vector<double>).
        /// @param mout Output matrix with scaled product (destination / input-output, Matrix<float>).
        template<typename Allocator, typename GPUAllocator, typename All2, typename GPUAll2>
        void MPSymMatrixMatrixMultiplication(const CudaRuntime& cudart, const MPRuntime& mp_rt, const Vector<double, Allocator, GPUAllocator>& d1, const MixedPrecisionMatrix<float, All2, GPUAll2>& m1,
                                             const Vector<double, Allocator, GPUAllocator>& d2, const MixedPrecisionMatrix<float, All2, GPUAll2>& m2,
                                             Vector<double, Allocator, GPUAllocator>& dout, Matrix<float, All2, GPUAll2>& mout)
        {
            
            if (mp_rt.fast_mode)
            {
                m1.resetSplit();
                m2.resetSplit();
                SymMatrixMatrixProduct(cudart, mp_rt, m1, m2, mout);
            }
            else
            {
                MatrixMatrixProduct(cudart, m1, m2, mout);
            }
            HadamardProduct(cudart, d1, d2, dout);
            SymmetrizedON2ScalingProductGPU(cudart, d1, m1, d2, m2, mout);
            SymmetrizeMatrix(cudart, mout);
            
        };
    

       
    } // namespace gpu 
} // namespace lahva