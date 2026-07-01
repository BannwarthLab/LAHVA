# GPU tests

gpu_test_utils_dep = declare_dependency(
  link_with: test_utils_lib,
  include_directories: include_directories('../common'),
)

# GPU Tensor type tests
test('gpu-tensor-types',
  executable(
    'lahva-gpu-tensor-types-test',
    sources: files('types/gpu-tensor-types.cpp'),
    dependencies: [lahva_dep, gpu_test_utils_dep],
    link_language: 'cpp',
  ),
)

# Vector type tests
test('gpu-types-vector',
  executable(
    'lahva-gpu-vector-types-test',
    sources: files('types/vector-types.cpp'),
    dependencies: [lahva_dep, gpu_test_utils_dep],
    link_language: 'cpp',
  ),
)

# Matrix type tests
test('gpu-types-matrix',
  executable(
    'lahva-gpu-matrix-types-test',
    sources: files('types/matrix-types.cpp'),
    dependencies: [lahva_dep, gpu_test_utils_dep],
    link_language: 'cpp',
  ),
)

# LowTriMatrix type tests
test('gpu-types-lowtrimatrix',
  executable(
    'lahva-gpu-lowtrimatrix-types-test',
    sources: files('types/lowtrimatrix-types.cpp'),
    dependencies: [lahva_dep, gpu_test_utils_dep],
    link_language: 'cpp',
  ),
)

# Mixed-Precision Matrix tests
test('gpu-mixed-precision-matrix',
  executable(
    'lahva-gpu-mixed-precision-matrix-test',
    sources: files('types/mixed-precision-matrix-test.cpp'),
    dependencies: [lahva_dep, gpu_test_utils_dep],
    link_language: 'cpp',
  ),
)

# GPU miscellaneous tests
test('gpu-misc',
  executable(
    'lahva-gpu-misc-test',
    sources: files('utils/gpu-misc.cpp'),
    dependencies: [lahva_dep, gpu_test_utils_dep],
    link_language: 'cpp',
  ),
  is_parallel : false
)

# BLAS Level 1 tests
test('gpu-blas-level1',
  executable(
    'lahva-gpu-level1-test',
    sources: files('blas/level1.cpp'),
    dependencies: [lahva_dep, gpu_test_utils_dep],
    link_language: 'cpp',
  ),
)

# BLAS Level 2 tests
test('gpu-blas-level2',
  executable(
    'lahva-gpu-level2-test',
    sources: files('blas/level2.cpp'),
    dependencies: [lahva_dep, gpu_test_utils_dep],
    link_language: 'cpp',
  ),
)

# BLAS Level 3 tests
test('gpu-blas-level3',
  executable(
    'lahva-gpu-level3-test',
    sources: files('blas/level3.cpp'),
    dependencies: [lahva_dep, gpu_test_utils_dep],
    link_language: 'cpp',
  ),
)

# Additional Level 1 tests
test('gpu-additional-level1',
  executable(
    'lahva-gpu-additional-level1-test',
    sources: files('additional/level1.cpp'),
    dependencies: [lahva_dep, gpu_test_utils_dep],
    link_language: 'cpp',
  ),
)

# Additional Level 2 tests
test('gpu-additional-level2',
  executable(
    'lahva-gpu-additional-level2-test',
    sources: files('additional/level2.cpp'),
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
  timeout: 60,
)
