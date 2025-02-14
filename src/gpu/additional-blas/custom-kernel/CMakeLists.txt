set(dir ${CMAKE_CURRENT_SOURCE_DIR})

# Add source files
list(APPEND SRC_FILES
  ${dir}/hadamard.cu
  ${dir}/mixed-precision.cu
  ${dir}/norm.cu
  ${dir}/trace.cu
)

set(SRC_FILES ${SRC_FILES} PARENT_SCOPE)