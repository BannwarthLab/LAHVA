# Common test utilities library
test_utils_lib = static_library(
  'test-utils',
  files('utils.cpp'),
  dependencies: lahva_dep,
  include_directories: include_directories('.'),
  link_language: 'cpp',
)

test_utils_dep = declare_dependency(
  link_with: test_utils_lib,
  include_directories: include_directories('.'),
)
