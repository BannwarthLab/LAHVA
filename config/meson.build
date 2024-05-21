# This file is part of tblite.
# SPDX-Identifier: LGPL-3.0-or-later
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

omp_dep = dependency('openmp')
lib_deps += omp_dep

lapack_vendor = get_option('lapack')
if lapack_vendor == 'auto'
  if cc_id == 'intel'
    lapack_vendor = 'mkl'
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


