fs = import('fs')
gpu = get_option('gpu')
debug = get_option('buildtype')
os = host_machine.system()
c = meson.get_compiler('c')
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

stdcpp = c.find_library('stdc++')
lib_deps += stdcpp

code = '''
  extern "C" {void dgetrf_(
    int const* m, int const* n,
    double* A, int const* lda, int* ipiv,
    int* info );}
  int main() { 
  int n = 2;
  int *ipiv = new int[n];
  int info = 0;
  double* a = new double[n*n]; 
  dgetrf_(&n, &n, a, &n, ipiv, &info); }
  '''


omp_dep = dependency('openmp', language: 'cpp', required: false)

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

  if cc_id == 'intel-llvm'
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
  if cc_id == 'intel' or cc_id == 'intel-llvm'
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
  
  if fs.exists('/usr/include/openblas')
    inc_dirs += include_directories('/usr/include/openblas')
  endif
  if fs.exists('/usr/include/x86_64-linux-gnu/openblas')
    inc_dirs += include_directories('/usr/include/x86_64-linux-gnu/openblas')
  endif
  if not cc.links(code, dependencies: openblas_dep, name: 'lapack_dsytrs', args: ['-llapack'] )
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
  
  blas_dep = dependency('blas', required: false, method : 'cmake')
  if not blas_dep.found()
    blas_dep = dependency('blas', required: false, method : 'pkg-config')
  endif
  if not blas_dep.found()
    blas_dep = dependency('blas', required: false, method : 'system')
  endif
  if not blas_dep.found()
    if fs.exists('/usr/include/cblas')
      include_directories('/usr/include/cblas')
    endif
    if fs.exists('/usr/include/x86_64-linux-gnu/cblas')
      include_directories('/usr/include/x86_64-linux-gnu/cblas')
    endif
    blas_dep = cc.find_library('blas')
  endif
  lib_deps += blas_dep
  if not cc.links(code, dependencies: blas_dep)
    lapack_dep = dependency('lapack', required: false, method : 'cmake')
    if not lapack_dep.found()
      lapack_dep = dependency('lapack', required: false, method : 'pkg-config')
    endif
    if not lapack_dep.found()
      lapack_dep = dependency('lapack', required: false, method : 'system')
    endif
  
    if not lapack_dep.found()
      if fs.exists('/usr/include/lapack')
        include_directories('/usr/include/lapack')
      endif
      if fs.exists('/usr/include/x86_64-linux-gnu/lapack')
        include_directories('/usr/include/x86_64-linux-gnu/lapack')
      endif
      lapack_dep = cc.find_library('lapack')
    endif
    lib_deps += lapack_dep
  endif
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

if cc.has_header('cblas/cblas.h')
  message('CBLAS header found')
  add_project_arguments(['-DCBLAS_HEADER'], language: 'cpp')
endif
if gpu == true
  arches = get_option('gpu_arch')
  has_nvcc = add_languages('cuda', native: false)
  nvcc = meson.get_compiler('cuda')
  cublas = nvcc.find_library('cublas')
  cusolver = nvcc.find_library('cusolver')
  cusparse = nvcc.find_library('cusparse')
  lib_deps += cublas
  lib_deps += cusolver
  lib_deps += cusparse
  cuda = dependency('cuda', modules: ['cudart'])
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
  add_project_arguments('-DNDEBUG', language: 'cpp')
  if gpu == true
    add_project_arguments('-DNDEBUG', language: 'cuda')
  endif
endif
