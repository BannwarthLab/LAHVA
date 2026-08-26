set(dir ${CMAKE_CURRENT_SOURCE_DIR})

# Add source files
list(APPEND SRC_FILES
  ${dir}/cusolver_runtime.cpp
  ${dir}/cusparse_runtime.cpp
  ${dir}/runtime.cpp
  ${dir}/timer.cpp
  ${dir}/unpack-vector.cu
  ${dir}/utils.cpp
)

set(SRC_FILES ${SRC_FILES} PARENT_SCOPE)