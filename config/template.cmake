@PACKAGE_INIT@

set("@PROJECT_NAME@_WITH_OpenMP" @WITH_OpenMP@)
set("@PROJECT_NAME@_WITH_API" @WITH_API@)
set(
  "@PROJECT_NAME@_INCLUDE_DIRS"
  "@CMAKE_INSTALL_PREFIX@/@CMAKE_INSTALL_INCLUDEDIR@"
  "@CMAKE_INSTALL_PREFIX@/@CMAKE_INSTALL_INCLUDEDIR@/@module-dir@"
)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")

if(NOT TARGET "@PROJECT_NAME@::@PROJECT_NAME@")
  include("${CMAKE_CURRENT_LIST_DIR}/@PROJECT_NAME@-targets.cmake")

  include(CMakeFindDependencyMacro)

  if(NOT TARGET "OpenMP::OpenMP_CXX")
    find_dependency("OpenMP")
  endif()

  if(NOT TARGET "LAPACK::LAPACK")
    find_dependency("LAPACK")
  endif()

  if("@PROJECT_NAME@_GPU")
    find_dependency("CUDAToolkit")
  endif()
 
endif()
