
cmake_minimum_required(VERSION 3.18)

project(
  "lahva"
  LANGUAGES "C" "CXX"
  VERSION "0.1.0"
  DESCRIPTION "Linear Algebra on Heterogenous/Vectorized Architecture"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(WITH_TESTS "Copmile Testsuite" ON)
# Follow GNU conventions for installing directories
include(GNUInstallDirs)

option(GPU "Enable GPU support" OFF)
option(LAPACK_VENDOR "Specify LAPACK vendor" "auto")
option(CUSTOM_LIBRARIES "Custom libraries for LAPACK" "")
#option(CUDA_ARCHITECTURES "NVIDIA Compute Capability (list of strings)" "89;86;80")

option(WITH_TESTS "Copmile Testsuite" OFF)

add_library(
  "${PROJECT_NAME}-lib" SHARED
)

file(GLOB_RECURSE LAHVA_HEADER "include/*.h*")

set_target_properties(
  "${PROJECT_NAME}-lib"
  PROPERTIES
  POSITION_INDEPENDENT_CODE TRUE
  OUTPUT_NAME "${PROJECT_NAME}"
  VERSION "${PROJECT_VERSION}"
  SOVERSION "${PROJECT_VERSION_MAJOR}"
  WINDOWS_EXPORT_ALL_SYMBOLS TRUE
  PUBLIC_HEADER "${LAHVA_HEADER}"
  #Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include"
)
set(PRE_FLAGS "")
# General configuration information
add_subdirectory("config")

if(GPU)
  if(CMAKE_CUDA_ARCHITECTURES)
    set_property(TARGET "${PROJECT_NAME}-lib" PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
  else()
    set_property(TARGET "${PROJECT_NAME}-lib" PROPERTY CUDA_ARCHITECTURES native)
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    find_program(CC_BIN ${CMAKE_CXX_COMPILER})
    target_compile_options(${PROJECT_NAME}-lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    -qopenmp
    --allow-unsupported-compiler 
    -ccbin 
    ${CC_BIN}
    >)
  endif()

  target_link_libraries("${PROJECT_NAME}-lib" PUBLIC "CUDA::cudart" "CUDA::cublas" "CUDA::cusolver" "CUDA::cusparse")
  list(APPEND PRE_FLAGS "_CUDA")
endif()

# Collect source of the project
set(SRC_FILES)
set(HEADER_FILES)
add_subdirectory("src")

target_sources(
  "${PROJECT_NAME}-lib" PRIVATE
  ${SRC_FILES}
)

# LAHVA library target
add_compile_definitions(${PRE_FLAGS})

message(STATUS ${CMAKE_CURRENT_SOURCE_DIR})

if (GPU)
  message(STATUS ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(
  "${PROJECT_NAME}-lib"
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/additional-blas>
  $<INSTALL_INTERFACE:include>
)

else()
  message(STATUS ${CMAKE_CURRENT_SOURCE_DIR})
  target_include_directories(
    "${PROJECT_NAME}-lib"
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
endif()
if(NOT EXISTS "${PROJECT_BINARY_DIR}/include")
  file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/include")
endif()


# Export targets for other projects
add_library("${PROJECT_NAME}" INTERFACE)
target_link_libraries("${PROJECT_NAME}" INTERFACE "${PROJECT_NAME}-lib")

install(
  TARGETS
  "${PROJECT_NAME}"
  "${PROJECT_NAME}-lib"
  EXPORT
  "${PROJECT_NAME}-targets"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PUBLIC_HEADER DESTINATION include
)
install(
  EXPORT
  "${PROJECT_NAME}-targets"
  NAMESPACE
  "${PROJECT_NAME}::"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
)
install(
  DIRECTORY
  "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
install(
  DIRECTORY
  "${CMAKE_CURRENT_BINARY_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${module-dir}"
)
export(TARGETS
  "${PROJECT_NAME}"
  "${PROJECT_NAME}-lib"
  NAMESPACE
  lahva::
  FILE
  "${PROJECT_NAME}-targets.cmake"
)

# add the testsuite
if (WITH_TESTS)
  enable_testing()
  add_subdirectory("test")
endif()