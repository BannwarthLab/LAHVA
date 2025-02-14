add_subdirectory(additional-blas)

# Add source files
list(APPEND SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/level1.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/level2.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/level3.cpp
)

set(SRC_FILES ${SRC_FILES} PARENT_SCOPE)