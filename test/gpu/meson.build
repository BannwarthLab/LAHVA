# GPU tests

gpu_test_utils_dep = declare_dependency(
  include_directories: include_directories('../common'),
)

# GPU Tensor type tests
test('gpu-tensor-tensor',
  executable(
    'lahva-gpu-tensor-tensor',
    sources: files('tensor/gpu-tensor.cpp'),
    dependencies: [lahva_dep, gpu_test_utils_dep],
    link_language: 'cpp',
  ),
)

# Vector type tests
test('gpu-tensor-vector',
  executable(
    'lahva-gpu-vector-tensor',
    sources: files('tensor/vector.cpp'),
    dependencies: [lahva_dep, gpu_test_utils_dep],
    link_language: 'cpp',
  ),
)

# Matrix type tests
test('gpu-tensor-matrix',
  executable(
    'lahva-gpu-matrix-tensor',
    sources: files('tensor/matrix.cpp'),
    dependencies: [lahva_dep, gpu_test_utils_dep],
    link_language: 'cpp',
  ),
)

# LowTriMatrix type tests
test('gpu-tensor-lowtrimatrix',
  executable(
    'lahva-gpu-lowtrimatrix-tensor',
    sources: files('tensor/lowtrimatrix.cpp'),
    dependencies: [lahva_dep, gpu_test_utils_dep],
    link_language: 'cpp',
  ),
)

# Mixed-Precision Matrix tests
test('gpu-tensor-mixed-precision-matrix',
  executable(
    'lahva-gpu-tensor-mixed-precision-matrix',
    sources: files('tensor/mixed-precision-matrix.cpp'),
    dependencies: [lahva_dep, gpu_test_utils_dep],
    link_language: 'cpp',
  ),
)

# Sparsematrix type tests
test('gpu-tensor-sparsematrix',
  executable(
    'lahva-gpu-tensor-sparsematrix',
    sources: files('tensor/sparse-matrix.cpp'),
    dependencies: [lahva_dep, gpu_test_utils_dep],
    link_language: 'cpp',
  ),
)

# Blockmatrix type tests
test('gpu-tensor-blockmatrix',
  executable(
    'lahva-gpu-tensor-blockmatrix',
    sources: files('tensor/block-matrix.cpp'),
    dependencies: [lahva_dep, gpu_test_utils_dep],
    link_language: 'cpp',
  ),
)

# Blockdiag type tests
test('gpu-tensor-blockdiagmatrix',
  executable(
    'lahva-gpu-tensor-blockdiagmatrix',
    sources: files('tensor/block-diag-matrix.cpp'),
    dependencies: [lahva_dep, gpu_test_utils_dep],
    link_language: 'cpp',
  ),
)

# GPU miscellaneous tests
test('gpu-utils-misc',
  executable(
    'lahva-gpu-utils-misc',
    sources: files('utils/gpu-misc.cpp'),
    dependencies: [lahva_dep, gpu_test_utils_dep],
    link_language: 'cpp',
  ),
  is_parallel : false
)

# BLAS Level 1 tests
test('gpu-blas-level1',
  executable(
    'lahva-gpu-level1',
    sources: files('blas/level1.cpp'),
    dependencies: [lahva_dep, gpu_test_utils_dep],
    link_language: 'cpp',
  ),
)

# BLAS Level 2 tests
test('gpu-blas-level2',
  executable(
    'lahva-gpu-level2',
    sources: files('blas/level2.cpp'),
    dependencies: [lahva_dep, gpu_test_utils_dep],
    link_language: 'cpp',
  ),
)

# BLAS Level 3 tests
test('gpu-blas-level3',
  executable(
    'lahva-gpu-level3',
    sources: files('blas/level3.cpp'),
    dependencies: [lahva_dep, gpu_test_utils_dep],
    link_language: 'cpp',
  ),
)

# Additional Level 1 tests
test('gpu-blas-additional-level1',
  executable(
    'lahva-gpu-blas-additional-level1',
    sources: files('blas/additional-level1.cpp'),
    dependencies: [lahva_dep, gpu_test_utils_dep],
    link_language: 'cpp',
  ),
)

# Additional Level 2 tests
test('gpu-blas-additional-level2',
  executable(
    'lahva-gpu-additional-level2',
    sources: files('blas/additional-level2.cpp'),
    dependencies: [lahva_dep, gpu_test_utils_dep],
    link_language: 'cpp',
  ),
)

# GPU mixed precision tests
test('gpu-mixed-precision',
  executable(
    'lahva-gpu-test-mp',
    sources: files('utils/test-mp.cpp'),
    dependencies: [lahva_dep, gpu_test_utils_dep],
    link_language: 'cpp',
  ),
  timeout: 120,
)