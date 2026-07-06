![LAHVA-Logo](./graphics/lahva_logo.jpg)

# LAHVA - Linear Algebra on Heterogeneous/Vectorized Architecture

LAHVA is a modern C++ library that lets you write GPU-accelerated linear algebra code without learning CUDA. If you're familiar with BLAS and LAPACK on CPU, you already know LAHVA's API—it works the same way on GPU. Just create a `CudaRuntime` and pass it to your operations. The runtime handles all the GPU complexity: device memory, streams, handles, and synchronization. LAHVA even offers mixed-precision operations on the GPU to accelerate your code even more. 

## Key Features

- Unified API for CPU and GPU linear algebra operations
- Runtime abstraction managing GPU resources (streams, handles, memory) so you don't manage CUDA directly
- Tensor classes (Vector, Matrix, Lower Triangular Matrix) with template-based precision and execution location
- Automatic memory management with custom allocators for host and device memory spaces
- Mixed-precision support for GPU operations
- Straightforward switching between CPU and GPU execution via template parameters

## Motivation

You've written CPU linear algebra code using BLAS and LAPACK. Now you want to move it to GPU for speed. But GPU programming requires learning CUDA, cuBLAS, memory management, streams, and synchronization, in other words: a steep learning curve for what should be a straightforward port.

LAHVA bridges this gap. Its API is identical to CPU BLAS/LAPACK side you keep your algorithm unchanged. Instead of managing CUDA streams, handles, and device memory yourself, you create a single `CudaRuntime` object and pass it to your operations. The runtime handles all the GPU infrastructure: device selection, cuBLAS/cuSOLVER setup, memory transfers, and synchronization.

The iceberg graphic below illustrates this philosophy: you see a familiar BLAS/LAPACK interface while GPU resource management is completely hidden below.

![Graphical Motivation](./graphics/iceberg.png)

## Core Concepts

### The Runtime Abstraction

The **Runtime** is the central innovation in LAHVA. It encapsulates GPU resource management and provides a unified context for computations:

```cpp
// Create a GPU runtime - this handles all CUDA resource setup
lahva::CudaRuntime gpu_runtime;

// The runtime manages:
// - CUDA device selection (automatically picks GPU with max memory)
// - cuBLAS handle lifecycle
// - CUDA streams for asynchronous execution
// - Memory availability checking
// - cuSOLVER handle for advanced operations
```

When you execute linear algebra operations on GPU tensors, you pass the runtime as context. The runtime:
1. Manages device selection - automatically selects the GPU with the most available memory
2. Handles stream management - creates and manages CUDA streams for asynchronous execution
3. Encapsulates cuBLAS/cuSOLVER handles - library contexts are created/destroyed automatically
4. Performs memory checking - verifies sufficient GPU memory is available before operations
5. Synchronizes execution - provides methods to synchronize GPU operations when needed

This means you write linear algebra code without touching CUDA directly - the runtime abstracts away all the boilerplate.

### Tensor Classes

LAHVA provides three tensor abstractions for both CPU and GPU:

- **Vector** - 1D tensor for vectors
- **Matrix** - 2D tensor for matrices  
- **LowerTriMatrix** - Symmetric 2D tensor stored in packed format

All tensor classes:
- Support multiple numerical precisions via template parameters (`int`, `float`, `double`, `__half` on GPU)
- Use allocators for memory management (similar to `std::vector`)
- Have both CPU-only and GPU-capable variants
- Support RAII semantics for automatic memory cleanup

For GPU tensors, memory is split between host and device spaces:
```cpp
// GPU Matrix with explicit allocators for host and device memory
lahva::gpu::Matrix<float, CudaHostAllocator<float>, CudaDeviceAsyncAllocator<float>> mat(5, 5, 1.0);
// mat automatically manages transfer between host and device memory
```

### Static vs. Polymorphic Usage

**Static Usage** (fixed at compile time):
```cpp
#include <linalg.hpp>
using namespace lahva::cpu;
Vector<double> cpu_vec(5, 2.0);

using namespace lahva::gpu;
Vector<double> gpu_vec(5, 2.0);
```

**Polymorphic Usage** (runtime selection):
```cpp
#include "example/lahva_wrap.hpp"

template<typename blas_impl>
class MyClass {
  template<typename U>
  using Vector = typename TensorFactory<blas_impl>::template Vector<U>;
  template<typename U>
  using Matrix = typename TensorFactory<blas_impl>::template Matrix<U>;
  // ...
};

// Instantiate for CPU or GPU at runtime
MyClass<cpuBLAS> cpu_version;
MyClass<gpuBLAS> gpu_version;
```

### Mixed Precision
LAHVA provides mixed-precision linear algebra operations on NVIDIA GPUs, enabling computations where different numerical precisions are used within the same algorithm. Mixed precision leverages lower precision (FP16/half-precision) for faster computation and reduced memory usage while maintaining accuracy through selective use of higher precision (FP32/FP64) for critical operations. This approach is especially valuable on consumer-grade GPUs, which often have significantly higher throughput for FP16 operations compared to professional hardware. By using mixed precision, you can achieve:

LAHVA abstracts the complexity of managing mixed-precision matrices and ensuring numerical stability, allowing you to benefit from GPU acceleration without manually orchestrating precision conversions.

## Building from Source

LAHVA supports both Meson and CMake build systems.

### GPU Build (Nvidia)

#### Compile with Meson

```bash
meson setup _build -Dgpu=true -Dlapack=mkl  # or openblas
meson compile -C _build
meson test -C _build
```

Or for a CPU-only build:
```bash
meson setup _build -Dgpu=false
meson compile -C _build
meson test -C _build
```

#### Compile with CMake

```bash
mkdir build && cd build
cmake .. -DENABLE_GPU=ON
make
ctest
```

Or for a CPU-only build:
```bash
mkdir build && cd build
cmake .. -DENABLE_GPU=OFF
make
ctest
```

### Using LAHVA as a Subproject

To use LAHVA as a dependency in another Meson project:

```python
lahva_dep = dependency(
  'lahva',
  version: '>=0.2.0',
  fallback: ['lahva', 'lahva_dep'],
  default_options: ['default_library=static'],
)
```

Then link it to your target:
```python
executable('my_app', 'main.cpp', dependencies: [lahva_dep])
```

A more elaborate file to wrap LAHVA in your subproject is found in the examples [folder](examples/lahva_wrap.hpp).

## Usage Examples

### Basic CPU Usage

```cpp
#include <linalg.hpp>
#include <iostream>

using namespace lahva::cpu;

int main() {
    // Create vectors
    Vector<double> v1(5, 1.0);
    Vector<double> v2(5, 2.0);
    
    // Vector operations (BLAS Level 1)
    double dot_result = InnerVectorProduct(v1, v2);
    std::cout << "Dot product: " << dot_result << std::endl;
    
    Vector<double> v3(5, 0.0);
    AddVectors(1.0, v2, v3);  // v3 = v3 + 1.0 * v2
    
    // Create matrices
    Matrix<double> A(Shape(5, 5), 0.5);
    Matrix<double> B(Shape(5, 5), 1.5);
    Matrix<double> C(Shape(5, 5), 0.0);
    
    // Matrix-matrix multiplication (BLAS Level 3)
    MatrixMatrixProduct(A, B, C);  // C = A * B
    
    // Matrix-vector multiplication (BLAS Level 2)
    Vector<double> result(5, 0.0);
    MatrixVectorProduct("N", 1.0, A, v1, 1, 0.0, result, 1);  // result = A * v1
    
    return 0;
}
```

### Basic GPU Usage

```cpp
#include <linalg.hpp>
#include <iostream>

using namespace lahva::gpu;

int main() {
    // Create runtime - selects GPU with maximum available memory
    CudaRuntime cudart;
    
    // Create GPU tensors
    Vector<double> v1(100, 1.0);
    Vector<double> v2(100, 2.0);
    Matrix<float> A(Shape(50, 50), 0.5);
    Matrix<float> B(Shape(50, 50), 1.5);
    Matrix<float> C(Shape(50, 50), 0.0);
    
    // Vector operations (BLAS Level 1) - automatically on GPU
    double dot_result = InnerVectorProduct(v1, v2);
    
    // Matrix multiplication (BLAS Level 3) - accelerated by cuBLAS
    MatrixMatrixProduct(cudart, A, B, C);  // C = A * B
    
    // All memory transfers and cuBLAS calls are handled transparently by the runtime
    
    return 0;
}
```

## Architecture Details

### GPU Memory Management

GPU tensors handle memory in two separate spaces:

1. Host Memory - accessible by CPU, transferred from device
2. Device Memory - on GPU, accessed by cuBLAS/cuSOLVER kernels

LAHVA provides allocators for both spaces:
- `CudaHostAllocator<T>` - allocates pinned host memory for efficient GPU transfer
- `CudaDeviceAllocator<T>` - allocates device memory
- `CudaDeviceAsyncAllocator<T>` - allocates with asynchronous copy support

### Stream Management

The runtime manages CUDA streams for asynchronous execution:

```cpp
// Default: uses per-thread stream (implicit synchronization)
CudaRuntime default_runtime;

// Enable explicit stream management
default_runtime.enableAsyncCopy();

// Launch operations asynchronously
// ... GPU operations ...

// Synchronize when needed
default_runtime.synchronize();
```

## Testing & Compatibility

LAHVA has a comprehensive continuous integration test suite running on multiple OS, compiler, and BLAS library combinations. All tests are executed with both **Meson and CMake** build systems. Our GitLab CI/CD pipeline is run on an NVIDIA GeForce RTX 3070 but we occasionally test LAHVA on other GPUs as well. 


- Operating Systems: Rocky Linux 9, 10
- Compilers: GCC, Intel oneAPI (2023.2, 2025.3)
- BLAS Libraries: OpenBLAS, Intel MKL (2023.2, 2025.3)
- CUDA Versions: 11.8, 12.5, 13.0 (Rocky Linux 9); 13.0 (Rocky Linux 10)

> **Note:** LAHVA GPU support is NVIDIA-only using CUDA, cuBLAS, and cuSOLVER. There are no plans to extend support to other GPU manufacturers (such as AMD and Intel).

## Known Usage and Community

LAHVA originates from computational chemists, where GPU-accelerated linear algebra is essential for electronic structure calculations. 

However, LAHVA's design is domain-agnostic and can benefit any field requiring efficient GPU-accelerated linear algebra operations. LAHVA's simplified API and runtime abstraction can help you leverage GPU acceleration without dealing with low-level CUDA complexity. 

In case you are using LAHVA in a different field, please reach out via a [GitLab issue](https://git.rwth-aachen.de/bannwarthlab/lahva/-/issues). We're interested in understanding how LAHVA can serve the broader scientific computing community and would be happy to discuss how it might fit your use case. 

So far, we know that LAHVA is used in:
- [GAMBITS](https://git.rwth-aachen.de/bannwarthlab/gambits) - Purification methods for quantum chemistry

## Authors

- Pit Steinbach
- Mark Heezen
- Christoph Bannwarth

## License

LAHVA is distributed under the MIT License. See [LICENSE](LICENSE) for details.

## Project Status

LAHVA is actively maintained. The API is considered stable, though enhancements and bug fixes are continuously integrated. The library is suitable for production use with proper testing for your specific use case.

## Support & Issues

Please open an issue on GitLab if you encounter problems or have questions. We're happy to help with:
- Build issues
- CUDA/hardware compatibility questions  
- Feature requests
- Bug reports