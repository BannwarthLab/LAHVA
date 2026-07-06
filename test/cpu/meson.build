# CPU tests

cpu_test_utils_dep = declare_dependency(
  link_with: test_utils_lib,
  include_directories: include_directories('../common'),
)

# Vector type tests
test('cpu-tensor-vector',
  executable(
    'lahva-cpu-tensor-vector',
    sources: files('tensor/vector.cpp'),
    dependencies: [lahva_dep, cpu_test_utils_dep],
    link_language: 'cpp',
  ),
)

# Matrix type tests
test('cpu-tensor-matrix',
  executable(
    'lahva-cpu-matrix-tensor',
    sources: files('tensor/matrix.cpp'),
    dependencies: [lahva_dep, cpu_test_utils_dep],
    link_language: 'cpp',
  ),
)

# LowTriMatrix type tests
test('cpu-tensor-lowtrimatrix',
  executable(
    'lahva-cpu-lowtrimatrix-tensor',
    sources: files('tensor/lowtrimatrix.cpp'),
    dependencies: [lahva_dep, cpu_test_utils_dep],
    link_language: 'cpp',
  ),
)

# BLAS Level 1 tests
test('cpu-blas-level1',
  executable(
    'lahva-cpu-level1',
    sources: files('blas/level1.cpp'),
    dependencies: [lahva_dep, cpu_test_utils_dep],
    link_language: 'cpp',
  ),
)

# BLAS Level 2 tests
test('cpu-blas-level2',
  executable(
    'lahva-cpu-level2',
    sources: files('blas/level2.cpp'),
    dependencies: [lahva_dep, cpu_test_utils_dep],
    link_language: 'cpp',
  ),
)

# BLAS Level 3 tests
test('cpu-blas-level3',
  executable(
    'lahva-cpu-level3',
    sources: files('blas/level3.cpp'),
    dependencies: [lahva_dep, cpu_test_utils_dep],
    link_language: 'cpp',
  ),
)

# Additional Level 1 tests
test('cpu-blas-additional-level1',
  executable(
    'lahva-cpu-blas-additional-level1',
    sources: files('blas/additional-level1.cpp'),
    dependencies: [lahva_dep, cpu_test_utils_dep],
    link_language: 'cpp',
  ),
)

# Additional Level 2 tests
test('cpu-blas-additional-level2',
  executable(
    'lahva-cpu-blas-additional-level2',
    sources: files('blas/additional-level2.cpp'),
    dependencies: [lahva_dep, cpu_test_utils_dep],
    link_language: 'cpp',
  ),
)

# LAPACK tests
test('cpu-blas-lapack',
  executable(
    'lahva-cpu-blas-lapack',
    sources: files('blas/lapack.cpp'),
    dependencies: [lahva_dep, cpu_test_utils_dep],
    link_language: 'cpp',
  ),
)


