/// @file linalg.hpp
/// @brief Linear algebra tensor abstractions including vectors, matrices, and specialized matrix types.
///
/// This header aggregates tensor class declarations for both CPU and GPU computation.

#ifndef LAHVA_LINALG_H
#define LAHVA_LINALG_H
#include <memory>
#include <vector>

#include "impl/tensor/allocators.hpp"
#include "impl/tensor/cpu/blockdiagmatrix.hpp"
#include "impl/tensor/cpu/lowtrimatrix.hpp"
#include "impl/tensor/cpu/matrix.hpp"
#include "impl/tensor/cpu/tensor.hpp"
#ifdef _CUDA
#include <cuda_fp16.h>
#include "impl/mp-runtime.hpp"
#include "impl/tensor/gpu/blockdiagmatrix.hpp"
#include "impl/tensor/gpu/lowtrimatrix.hpp"
#include "impl/tensor/gpu/matrix.hpp"
#include "impl/tensor/gpu/vector.hpp"
#include "impl/tensor/gpu/matrix.hpp"
#include "impl/tensor/gpu/lowtrimatrix.hpp"
#include "impl/tensor/gpu/blockdiagmatrix.hpp"
#include "impl/tensor/gpu/mixed-precision-classes/mixed-precision-matrix.hpp"
#include "impl/mp-runtime.hpp"

#endif
#include "impl/tensor/cpu/vector.hpp"
#include "impl/tensor/cpu/matrix.hpp"
#include "impl/tensor/cpu/lowtrimatrix.hpp"
#include "impl/tensor/cpu/blockdiagmatrix.hpp"
#include "impl/tensor/allocators.hpp"

#endif // LAHVA_LINALG_H