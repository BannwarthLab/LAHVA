project(
  'lahva',
  'c',
  'cpp',
  
  version: '0.2.0',
  license: 'MIT',
  meson_version: '>=1.3.0',
  default_options: [
    'buildtype=debugoptimized',
    'default_library=both',
    'cpp_std=c++17',
    'c_std=c11',
    'cuda_std=c++17',
  ],
)
install = not (meson.is_subproject() and get_option('default_library') == 'static')

inc_dirs = []
lib_deps = []
subdir('config')

inc_dirs += include_directories('include') 
if has_gpu == true
  inc_dirs += include_directories('src/gpu/additional-blas', is_system: meson.is_subproject())
endif


# Collect source of the project
srcs = []
subdir('src')

# Library target
lahva_lib = library(
  meson.project_name(),
  sources: srcs,
  version: meson.project_version(),
  dependencies: lib_deps,
  include_directories: inc_dirs,
  install: install,
  link_language: 'cpp',
)
lahva_dep = declare_dependency(
  link_with: lahva_lib,
  include_directories: inc_dirs,
  dependencies: lib_deps,
  variables: {'includedir': meson.current_source_dir() / 'include'},
)

install = not (meson.is_subproject() and get_option('default_library') == 'static')

subdir('test')
subdir('examples')