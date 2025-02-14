add_subdirectory(custom-kernel)

set(dir ${CMAKE_CURRENT_SOURCE_DIR})

# Add source files
list(APPEND SRC_FILES
  ${dir}/additional-level1.cpp
  ${dir}/additional-level1.cu
  ${dir}/additional-level2.cpp
)

set(SRC_FILES ${SRC_FILES} PARENT_SCOPE)