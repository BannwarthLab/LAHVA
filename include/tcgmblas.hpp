#ifndef TCGMBLAS_BLAS_HPP
#define TCGMBLAS_BLAS_HPP

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
#include "impl/blas/gpu/lapack.hpp"
#endif
#endif