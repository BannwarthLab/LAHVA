# CPU tests

cpu_test_utils_dep = declare_dependency(
  link_with: test_utils_lib,
  include_directories: include_directories('../common'),
)

# Vector type tests
test('cpu-types-vector',
  executable(
    'lahva-cpu-vector-types-test',
    sources: files('types/vector-types.cpp'),
    dependencies: [lahva_dep, cpu_test_utils_dep],
    link_language: 'cpp',
  ),
)

# Matrix type tests
test('cpu-types-matrix',
  executable(
    'lahva-cpu-matrix-types-test',
    sources: files('types/matrix-types.cpp'),
    dependencies: [lahva_dep, cpu_test_utils_dep],
    link_language: 'cpp',
  ),
)

# LowTriMatrix type tests
test('cpu-types-lowtrimatrix',
  executable(
    'lahva-cpu-lowtrimatrix-types-test',
    sources: files('types/lowtrimatrix-types.cpp'),
    dependencies: [lahva_dep, cpu_test_utils_dep],
    link_language: 'cpp',
  ),
)

# BLAS Level 1 tests
test('cpu-blas-level1',
  executable(
    'lahva-cpu-level1-test',
    sources: files('impl/level1.cpp'),
    dependencies: [lahva_dep, cpu_test_utils_dep],
    link_language: 'cpp',
  ),
)

# BLAS Level 2 tests
test('cpu-blas-level2',
  executable(
    'lahva-cpu-level2-test',
    sources: files('impl/level2.cpp'),
    dependencies: [lahva_dep, cpu_test_utils_dep],
    link_language: 'cpp',
  ),
)

# BLAS Level 3 tests
test('cpu-blas-level3',
  executable(
    'lahva-cpu-level3-test',
    sources: files('impl/level3.cpp'),
    dependencies: [lahva_dep, cpu_test_utils_dep],
    link_language: 'cpp',
  ),
)

# Additional Level 1 tests
test('cpu-additional-level1',
  executable(
    'lahva-cpu-additional-level1-test',
    sources: files('blas/additional/level1.cpp'),
    dependencies: [lahva_dep, cpu_test_utils_dep],
    link_language: 'cpp',
  ),
)

# Additional Level 2 tests
test('cpu-additional-level2',
  executable(
    'lahva-cpu-additional-level2-test',
    sources: files('blas/additional/level2.cpp'),
    dependencies: [lahva_dep, cpu_test_utils_dep],
    link_language: 'cpp',
  ),
)

# LAPACK tests
test('cpu-lapack',
  executable(
    'lahva-cpu-lapack-test',
    sources: files('lapack-test.cpp'),
    dependencies: [lahva_dep, cpu_test_utils_dep],
    link_language: 'cpp',
  ),
)


