# Options
option(GPU "Enable GPU support" OFF)
option(LAPACK_VENDOR "BLAS/LAPCK Vendor")
set(CUSTOM_LIBRARIES "")
option(WITH_TESTS "Copmile Testsuite" ON)

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type")

# Compiler-specific arguments
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
# No additional arguments for GCC
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  add_compile_options(-traceback -static-libgcc)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "PGI" OR CMAKE_CXX_COMPILER_ID MATCHES "NVIDIA")
  add_compile_options(-Mbackslash -Mallocatable=03 -traceback)
endif()

# Dependencies
# find_library(STDCPP_LIB stdc++)
# list(APPEND lib_deps ${STDCPP_LIB})
find_package(OpenMP REQUIRED)
target_link_libraries("${PROJECT_NAME}-lib" PUBLIC "OpenMP::OpenMP_CXX")

if(OpenMP_FOUND)
  list(APPEND lib_deps OpenMP::OpenMP_CXX)
elseif(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  add_compile_options(-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include)
  link_libraries(-lomp -L/opt/homebrew/opt/libomp/lib)
endif()
if(LAPACK_VENDOR MATCHES OFF)
  set(LAPACK_VENDOR "auto")
endif()
message(STATUS ${LAPACK_VENDOR})
# LAPACK vendor selection
if(LAPACK_VENDOR MATCHES "auto")
  if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    set(LAPACK_VENDOR "mkl")
  elseif(APPLE)
    set(LAPACK_VENDOR "apple")
    list(APPEND PRE_FLAGS "_APPLE")
  endif()
endif()

if(LAPACK_VENDOR STREQUAL "mkl")
  list(APPEND PRE_FLAGS "W_MKL")
  set(BLA_VENDOR Intel10_64lp)
  find_package(BLAS)
  target_link_libraries("${PROJECT_NAME}-lib" PUBLIC "BLAS::BLAS")
elseif(LAPACK_VENDOR MATCHES "openblas")
  set(BLA_VENDOR OpenBLAS)
  find_package(BLAS)
  target_link_libraries("${PROJECT_NAME}-lib" PUBLIC "BLAS::BLAS")

elseif(LAPACK_VENDOR STREQUAL "custom")
  foreach(lib IN LISTS CUSTOM_LIBRARIES)
    find_library(CUSTOM_LIB ${lib})
    list(APPEND lib_deps ${CUSTOM_LIB})
  endforeach()

elseif(LAPACK_VENDOR STREQUAL "apple")
  link_libraries(-framework Accelerate)

else()
  find_package(LAPACK REQUIRED)

  if(LAPACK_FOUND)
    target_link_libraries("${PROJECT_NAME}-lib" PUBLIC "LAPACK::LAPACK")
  else()
    find_library(LAPACK_LIB lapack)
    list(APPEND lib_deps ${LAPACK_LIB})
  endif()

  find_package(BLAS REQUIRED GLOBAL)
  if("${BLAS_LIBRARIES}" MATCHES "intel")
    list(APPEND PRE_FLAGS "W_MKL")
  endif()
  if(BLAS_FOUND)
    target_link_libraries("${PROJECT_NAME}-lib" PUBLIC "BLAS::BLAS")
  else()
    find_library(BLAS_LIB blas)
    list(APPEND lib_deps ${BLAS_LIB})
  endif()
endif()

# Release build type
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  list(APPEND PRE_FLAGS "NDEBUG")
endif()

set(PRE_FLAGS ${PRE_FLAGS} PARENT_SCOPE)