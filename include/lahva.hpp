/// @file lahva.hpp
/// @brief C++-style BLAS and LAPACK function declarations for CPU and GPU computation.
///
/// This header aggregates all C++-style function declarations (templates and overloads)
/// for BLAS Level 1, Level 2, Level 3, additional levels, and LAPACK routines.

#ifndef LAHVA_BLAS_HPP

#define LAHVA_BLAS_HPP

#include "impl/blas/cpu/level1.hpp"
#include "impl/blas/cpu/level2.hpp"
#include "impl/blas/cpu/level3.hpp"
#include "impl/blas/cpu/additional-level1.hpp"
#include "impl/blas/cpu/additional-level2.hpp"
#include "impl/blas/cpu/lapack.hpp"
#ifdef _CUDA
#include "impl/blas/gpu/level1.hpp"
#include "impl/blas/gpu/level2.hpp"
#include "impl/blas/gpu/level3.hpp"
#include "impl/blas/gpu/additional-level1.hpp"
#include "impl/blas/gpu/additional-level2.hpp"
#include "impl/blas/gpu/additional-level3.hpp"
#endif
#endif