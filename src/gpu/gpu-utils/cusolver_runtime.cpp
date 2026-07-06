/// @file cusolver_runtime.cpp
/// @brief NVIDIA cuSolver runtime wrapper implementation.
///
/// Provides CUDA cuSolver library initialization, cleanup, and stream management
/// for GPU-accelerated linear algebra operations.

#include <iostream>
#include "runtime.hpp"

/// @brief Checks cuSolver error status and throws on failure.
///
/// Validates cuSolver API return codes and throws an exception with file/line information on error.
///
/// @param stat cuSolver status code to check.
/// @param file Source file where error check occurred.
/// @param line Line number where error check occurred.
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
    /// @brief Initializes cuSolver dense solver handle.
    ///
    /// Creates and initializes the cuSolver dense solver context for GPU linear algebra operations.
    cuSolverRuntime::cuSolverRuntime()
    {
        get_cusolv_error(cusolverDnCreate(&handle));
    };

    /// @brief Destroys cuSolver dense solver handle and frees resources.
    ///
    /// Cleans up the cuSolver context and releases associated GPU resources.
    cuSolverRuntime::~cuSolverRuntime()
    {
        get_cusolv_error(cusolverDnDestroy(handle));
    };

    /// @brief Associates cuSolver operations with a specific CUDA stream.
    ///
    /// Sets the CUDA stream for subsequent cuSolver operations to enable asynchronous execution.
    ///
    /// @param stream CUDA stream for cuSolver operations.
    void cuSolverRuntime::setStream(const cudaStream_t& stream)
    {
        get_cusolv_error(cusolverDnSetStream(handle, stream));
    }

    
} // namespace gpusolver
