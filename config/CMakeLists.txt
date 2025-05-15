# Options
option(GPU "Enable GPU support" OFF)
option(LAPACK_VENDOR "BLAS/LAPCK Vendor")
set(CUSTOM_LIBRARIES "")
option(WITH_TESTS "Copmile Testsuite" ON)

# Set build type as CMake does not provide defaults
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(
    CMAKE_BUILD_TYPE "RelWithDebInfo"
    CACHE STRING "Build type to be used."
    FORCE
  )
  message(
    STATUS
    "Setting build type to '${CMAKE_BUILD_TYPE}' as none was specified."
  )
endif()

# Compiler-specific arguments
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
# No additional arguments for GCC
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  list(APPEND COMPILER_FLAGS -traceback)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "PGI" OR CMAKE_CXX_COMPILER_ID MATCHES "NVIDIA")
  add_compile_options(-Mbackslash -Mallocatable=03 -traceback)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  list(APPEND COMPILER_FLAGS "/Za")
endif()

# Dependencies
find_package(OpenMP)

if(OpenMP_FOUND)
  target_link_libraries("${PROJECT_NAME}-lib" PUBLIC "OpenMP::OpenMP_CXX")
  list(APPEND PRE_FLAGS "W_OPENMP")
elseif(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  list(APPEND COMPILER_FLAGS "-Xpreprocessor" "-fopenmp" "-I/opt/homebrew/opt/libomp/include")
  target_link_libraries("${PROJECT_NAME}-lib" PUBLIC -lomp -L/opt/homebrew/opt/libomp/lib)
  list(APPEND PRE_FLAGS "W_OPENMP")
endif()

if(LAPACK_VENDOR MATCHES OFF)
  set(LAPACK_VENDOR "auto")
endif()

# LAPACK vendor selection
if(LAPACK_VENDOR MATCHES "auto")
  if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    set(LAPACK_VENDOR "mkl")
  elseif(APPLE)
    set(LAPACK_VENDOR "apple")
  endif()
endif()

if(LAPACK_VENDOR STREQUAL "mkl")
  list(APPEND PRE_FLAGS "W_MKL")
  set(BLA_VENDOR Intel10_64lp)
  find_package(BLAS)
  target_link_libraries("${PROJECT_NAME}-lib" PUBLIC "BLAS::BLAS")
  find_package(LAPACK )
  target_link_libraries("${PROJECT_NAME}-lib" PUBLIC "LAPACK::LAPACK")
elseif(LAPACK_VENDOR MATCHES "openblas")
  set(BLA_VENDOR OpenBLAS)
  find_package(BLAS REQUIRED)
  target_link_libraries("${PROJECT_NAME}-lib" PUBLIC "BLAS::BLAS")
  find_package(LAPACK REQUIRED)
  target_link_libraries("${PROJECT_NAME}-lib" PUBLIC "LAPACK::LAPACK")
elseif(LAPACK_VENDOR STREQUAL "custom")
  foreach(lib IN LISTS CUSTOM_LIBRARIES)
    find_library(CUSTOM_LIB ${lib})
    list(APPEND lib_deps ${CUSTOM_LIB})
  endforeach()

elseif(LAPACK_VENDOR STREQUAL "apple")
  # prefer openblas
  set(BLA_VENDOR OpenBLAS)
  find_package(BLAS)

  if(BLAS_FOUND)
    message(STATUS "Will use OpenBLAS over Accelerate")
  else()
    set(BLA_VENDOR Apple)
    list(APPEND PRE_FLAGS "_APPLE")
    find_package(BLAS REQUIRED)
    list(APPEND COMPILER_FLAGS "-flax-vector-conversions")
  endif()

  find_package(LAPACK REQUIRED)
  target_link_libraries("${PROJECT_NAME}-lib" PUBLIC "BLAS::BLAS")
  target_link_libraries("${PROJECT_NAME}-lib" PUBLIC "LAPACK::LAPACK")
else()
  find_package(LAPACK REQUIRED)
  find_package(BLAS REQUIRED)
  set(BLA_VENDOR custom)
  if(LAPACK_FOUND)
    target_link_libraries("${PROJECT_NAME}-lib" PUBLIC "LAPACK::LAPACK")
  else()
    find_library(LAPACK_LIB lapack)
    list(APPEND lib_deps ${LAPACK_LIB})
  endif()

  if("${BLAS_LIBRARIES}" MATCHES "intel")
    list(APPEND PRE_FLAGS "W_MKL")
    set(BLA_VENDOR Intel10_64lp)
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

#if accelerate is used the header will be found in the system include path
if (BLA_VENDOR MATCHES OpenBLAS OR BLA_VENDOR MATCHES custom)
get_filename_component(LAPACK_LIB "${LAPACK_LIBRARIES}" DIRECTORY)
get_filename_component(LAPACK_LIB "${LAPACK_LIB}" DIRECTORY)
find_file(LAPACK_HEADER_FILE "lapack.h" PATHS /usr/include /usr/local/include "${LAPACK_LIB}/include" "${LAPACK_LIB}/include/openblas")

if(LAPACK_HEADER_FILE)
  message(STATUS "Found header file: ${LAPACK_HEADER_FILE}")
  get_filename_component(LAPACK_HEADER_FILE "${LAPACK_HEADER_FILE}" DIRECTORY)
  target_include_directories("${PROJECT_NAME}-lib" PUBLIC "${LAPACK_HEADER_FILE}")
else()
  message(STATUS "Header file not found")
  list(APPEND PRE_FLAGS "NETLIB_HEADER")
endif()

get_filename_component(BLAS_LIB "${BLAS_LIBRARIES}" DIRECTORY)
get_filename_component(BLAS_LIB "${BLAS_LIB}" DIRECTORY)
find_file(CBLAS_HDR "cblas.h" /usr/include /usr/local/include "${BLAS_LIB}/include" "${BLAS_LIB}/include/openblas")

if(CBLAS_HDR)
  message(STATUS "Found header file: ${CBLAS_HDR}")
else()
  message(FATAL_ERROR "Header file not found")
endif()

get_filename_component(CBLAS_HDR "${CBLAS_HDR}" DIRECTORY)
target_include_directories("${PROJECT_NAME}-lib" PUBLIC "${CBLAS_HDR}")
endif()
set(PRE_FLAGS ${PRE_FLAGS} PARENT_SCOPE)
set(COMPILER_FLAGS ${COMPILER_FLAGS} PARENT_SCOPE)

if(NOT DEFINED "${PROJECT_NAME}-dependency-method")
  set(
    "${PROJECT_NAME}-dependency-method"
    "subproject" "cmake" "pkgconf" "fetch"
  )
endif()

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" PARENT_SCOPE)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
endif()

install(
  DIRECTORY
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/template.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMinorVersion
)
install(
  FILES
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
)

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/template.pc"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc"
  @ONLY
)
install(
  FILES
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig"
)
