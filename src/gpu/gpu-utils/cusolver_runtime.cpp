/// @file cusolver_runtime.cpp
/// @brief NVIDIA cuSolver runtime wrapper implementation.
///
/// Provides CUDA cuSolver library initialization, cleanup, and stream management
/// for GPU-accelerated linear algebra operations.

#include <iostream>
#include "runtime.hpp"

void get_cusolv_ERROR(cusolverStatus_t stat, const char *file, int line)
    {
        cusolverStatus_t err_ = (stat);
        if (stat != CUSOLVER_STATUS_SUCCESS)
        {
            std::cerr << "cuSolver Error: " << err_ << std::endl
                      << "In File: " << file << " at line: " << std::to_string(line) << std::endl;
            throw std::runtime_error("cuSolver Error");
        }
    }
namespace lahva
{
    cuSolverRuntime::cuSolverRuntime()
    {
        get_cusolv_error(cusolverDnCreate(&handle));   
    };

    cuSolverRuntime::~cuSolverRuntime()
    {
        get_cusolv_error(cusolverDnDestroy(handle));
    };

    void cuSolverRuntime::setStream(const cudaStream_t& stream)
    {
        get_cusolv_error(cusolverDnSetStream(handle, stream));
    }

    
} // namespace gpusolver
