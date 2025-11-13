add_subdirectory(blas)
add_subdirectory(additional-blas)
add_subdirectory(gpu-utils)
add_subdirectory(mixed-precision-instantion)

set(SRC_FILES ${SRC_FILES} PARENT_SCOPE)