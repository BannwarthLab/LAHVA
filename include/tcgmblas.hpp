#ifndef TCGMBLAS_BLAS_HPP
#define TCGMBLAS_BLAS_HPP

#include "impl/cpu/level1.hpp"
#include "impl/cpu/level2.hpp"
#include "impl/cpu/level3.hpp"
#include "impl/cpu/additional-level1.hpp"
#include "impl/cpu/additional-level2.hpp"
#include "impl/cpu/lapack.hpp"
#ifdef _CUDA
#include "impl/gpu/level1.hpp"
#include "impl/gpu/level2.hpp"
#include "impl/gpu/level3.hpp"
#include "impl/gpu/additional-level1.hpp"
#include "impl/gpu/additional-level2.hpp"
#include "impl/gpu/additional-level3.hpp"
#include "impl/gpu/lapack.hpp"
#endif
#endif