#include "runtime.hpp"
#include <iostream>
namespace tcgmtensor
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

    void get_cusolv_ERROR(cusolverStatus_t stat, const char *file, int line)
    {
        if (stat != CUSOLVER_STATUS_SUCCESS)
        {
            std::cerr << "cuSolver Error: " << stat << std::endl
                      << "In File: " << file << " at line: " << std::to_string(line) << std::endl;
            throw std::runtime_error("cuSolver Error");
        }
    }
} // namespace gpusolver
