#ifndef LAHVA_LINALG_H
#define LAHVA_LINALG_H

#include <memory>
#include <vector>
#include <iterator>
#include <assert.h>
#include <iostream>
#include "impl/tensor/cpu/tensor.hpp"
#ifdef _CUDA
#include <cuda_fp16.h>
#include "impl/tensor/gpu/vector.hpp"
#include "impl/tensor/gpu/matrix.hpp"
#include "impl/tensor/gpu/lowtrimatrix.hpp"

template class lahva::gpu::Vector<float>;
template class lahva::gpu::Vector<double>;
template class lahva::gpu::Matrix<float>;
template class lahva::gpu::Matrix<double>;

#endif
#include "impl/tensor/cpu/vector.hpp"
#include "impl/tensor/cpu/matrix.hpp"
#include "impl/tensor/cpu/lowtrimatrix.hpp"
#include "impl/tensor/allocators.hpp"

template class lahva::cpu::Vector<float>;
template class lahva::cpu::Vector<double>;
template class lahva::cpu::Matrix<float>;
template class lahva::cpu::Matrix<double>;

#endif // LAHVA_LINALG_H