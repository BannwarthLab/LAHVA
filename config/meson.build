# This file is part of tblite.
# SPDX-License-Identifier: LGPL-3.0-or-later
#
# tblite is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tblite is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tblite.  If not, see <https://www.gnu.org/licenses/>.
gpu = get_option('gpu')
debug = get_option('buildtype')
os = host_machine.system()
cc = meson.get_compiler('cpp')
cc_id = cc.get_id()

if cc_id == 'gcc'
  add_project_arguments(
    language: 'cpp',
  )
elif cc_id == 'intel'
  add_project_arguments(
    '-traceback',
    '-static-libgcc',
    language: 'cpp',
  )
elif cc_id == 'pgi' or cc_id == 'nvidia_hpc'
  add_project_arguments(
    '-Mbackslash',
    '-Mallocatable=03',
    '-traceback',
    language: 'cpp',
  )
endif

#lib_deps += omp_dep

cpp_dep = cc.find_library('stdc++')
lib_deps += cpp_dep

omp_dep = dependency('openmp', required: false)

if omp_dep.found() 
  lib_deps += omp_dep
elif build_machine.system() == 'darwin' and cc_id == 'clang'
  add_project_arguments(['-Xpreprocessor','-fopenmp'],language: 'cpp')
  add_project_arguments(['-I/opt/homebrew/opt/libomp/include'], language: 'cpp')
  add_project_link_arguments(['-lomp'],language: 'cpp')
  add_project_link_arguments(['-L/opt/homebrew/opt/libomp/lib'], language: 'cpp')
endif

lapack_vendor = get_option('lapack')
if lapack_vendor == 'auto'
  if cc_id == 'intel'
    lapack_vendor = 'mkl'
  endif
 
  if build_machine.system() == 'darwin'
    lapack_vendor = 'apple'
    
  endif
endif

if lapack_vendor == 'mkl'
  add_project_arguments(['-DW_MKL'], language: 'cpp')
  if gpu == true
    add_project_arguments(['-DW_MKL'], language:  'cuda')
  endif
  mkl_dep = []
  if cc_id == 'intel'
    mkl_dep += cc.find_library('mkl_intel_lp64')
    mkl_dep += cc.find_library('mkl_tbb_thread')
  elif cc_id == 'gcc'
    mkl_dep += cc.find_library('mkl_gf_lp64')
    mkl_dep += cc.find_library('mkl_gnu_thread')
  else
    error('MKL not supported for this compiler')
  endif
  
  mkl_dep += cc.find_library('mkl_core')
  lib_deps += mkl_dep

elif lapack_vendor == 'mkl-rt'
  mkl_dep = cc.find_library('mkl_rt')
  lib_deps += mkl_dep

elif lapack_vendor == 'openblas'
  openblas_dep = dependency('openblas', required: false)
  if not openblas_dep.found()
    openblas_dep = cc.find_library('openblas')
  endif
  lib_deps += openblas_dep
  if not cc.links('external dsytrs; call dsytrs(); end', dependencies: openblas_dep)
    lapack_dep = dependency('lapack', required: false)
    
    if not lapack_dep.found()
      lapack_dep = cc.find_library('lapack')
    endif
    lib_deps += lapack_dep
  endif

elif lapack_vendor == 'custom'
  foreach lib: get_option('custom_libraries')
    lib_deps += cc.find_library(lib)
  endforeach
elif lapack_vendor == 'apple'
  openblas_dep = dependency('openblas', required: false)
  if not openblas_dep.found()
    add_project_link_arguments(['-framework','Accelerate'], language: 'cpp')
    add_project_arguments(['-D_APPLE'],language: 'cpp') 
  else
    lib_deps += openblas_dep
  endif
  

else
  lapack_dep = dependency('lapack', required: false)
  if not lapack_dep.found()
    lapack_dep = cc.find_library('lapack')
  endif
  lib_deps += lapack_dep
  blas_dep = dependency('blas', required: false)
  if not blas_dep.found()
    blas_dep = cc.find_library('blas')
  endif
  lib_deps += blas_dep
endif
if lapack_vendor != 'mkl'
if cc.has_header('lapack.h')
  message('LAPACK header found')
else
  message('LAPACK not found, using netlib headers')
  add_project_arguments(['-DNETLIB_HEADER'], language: 'cpp')
  if gpu == true
    add_project_arguments(['-DNETLIB_HEADER'], language: 'cuda')
  endif
endif
endif

if gpu == true
  arches = get_option('gpu_arch')
  has_nvcc = add_languages('cuda', native: false)
  cuda = dependency('cuda', modules: ['cudart', 'cublas', 'cusolver', 'cusparse'])
  foreach arch : arches
    if arch == 'native'
      add_project_arguments(['-arch=@0@'.format(arch)], language: 'cuda')
    else
      add_project_arguments(['-gencode=arch=compute_@0@,code=sm_@1@'.format(arch, arch)], language: 'cuda')
    endif
  endforeach
  if cc_id != 'gcc'
    cc_bin = find_program(cc.cmd_array()[0])
    add_project_arguments(['--allow-unsupported-compiler', '-ccbin', cc_bin.full_path()], language: 'cuda')
  endif

  lib_deps += cuda
  add_project_arguments(['-D_CUDA'], language: ['cuda', 'cpp'])
endif

if debug == 'release'
  add_project_arguments('-DNDEBUG', language: ['cpp'])
  if gpu == true
    add_project_arguments('-DNDEBUG', language: 'cuda')
  endif
endif
