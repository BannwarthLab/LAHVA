/// @file cusparse_runtime.cpp
/// @brief NVIDIA cuSPARSE runtime wrapper implementation.
///
/// Provides CUDA cuSPARSE library initialization, cleanup, and stream management
/// for GPU-accelerated sparse matrix operations.

#include <iostream>
#include "runtime.hpp"

/// @brief Checks cuSPARSE error status and throws on failure.
///
/// Validates cuSPARSE API return codes and throws an exception with file/line information on error.
///
/// @param stat cuSPARSE status code to check.
/// @param file Source file where error check occurred.
/// @param line Line number where error check occurred.
void get_cusparse_ERROR(cusparseStatus_t stat, const char *file, int line)
{
    if (stat != CUSPARSE_STATUS_SUCCESS)
    {
        std::cerr << "cuSPARSE Error: " << stat << std::endl
                  << "In File: " << file << " at line: " << std::to_string(line) << std::endl;
        throw std::runtime_error("cuSPARSE Error");
    }
}

namespace lahva
{
    /// @brief Initializes cuSPARSE handle.
    ///
    /// Creates and initializes the cuSPARSE context for GPU sparse linear algebra operations.
    cuSparseRuntime::cuSparseRuntime()
    {
        get_cusparse_error(cusparseCreate(&handle));
    };

    /// @brief Destroys cuSPARSE handle and frees resources.
    ///
    /// Cleans up the cuSPARSE context and releases associated GPU resources.
    cuSparseRuntime::~cuSparseRuntime()
    {
        get_cusparse_error(cusparseDestroy(handle));
    };

    /// @brief Associates cuSPARSE operations with a specific CUDA stream.
    ///
    /// Sets the CUDA stream for subsequent cuSPARSE operations to enable asynchronous execution.
    ///
    /// @param stream CUDA stream for cuSPARSE operations.
    void cuSparseRuntime::setStream(const cudaStream_t& stream)
    {
        get_cusparse_error(cusparseSetStream(handle, stream));
    }

} // namespace lahva
