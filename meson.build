project(
  'tcgm-c-blas',
  'c',
  'cpp',
  
  version: '0.0.0',
  license: 'LGPL-3.0-or-later',
  meson_version: '>=0.57.2',
  default_options: [
    'buildtype=debugoptimized',
    'default_library=both',
    'cpp_std=c++17'
  ],
)
install = not (meson.is_subproject() and get_option('default_library') == 'static')

gpu = get_option('gpu')

lib_deps = []

if gpu == true
  has_nvcc = add_languages('cuda', native: false)
  lib_deps += dependency('cuda', modules: ['cudart', 'cublas'])
  add_project_arguments(['-D_CUDA'], language: 'cpp')
endif


inc_dirs = []
subdir('config')

inc_dirs += include_directories('include') 
inc_dirs += include_directories('include/impl/cpu')
if gpu == true
  inc_dirs += include_directories('include/impl/gpu')
endif

# Collect source of the project
srcs = []
subdir('src')

add_project_arguments(['-fPIE'],language: 'cpp')

#lib_deps += [dependency('boost'), dependency('cuda', version : '>=10', modules : ['cublas'])]
# Library target
tcgmblas_lib = library(
  meson.project_name(),
  sources: srcs,
  #version: meson.project_version(),
  dependencies: lib_deps,
  include_directories: inc_dirs,
  install: install,
)
tcgmblas_dep = declare_dependency(
  link_with: tcgmblas_lib,
  include_directories: inc_dirs,
  dependencies: lib_deps,
  variables: {'includedir': meson.current_source_dir() / 'include'},
  
)

install = not (meson.is_subproject() and get_option('default_library') == 'static')
subdir('test')
