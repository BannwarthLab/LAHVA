set(dir ${CMAKE_CURRENT_SOURCE_DIR})


# Add source files
list(APPEND SRC_FILES
  ${dir}/level1.cpp
  ${dir}/level2.cpp
  ${dir}/level3.cpp
)

set(SRC_FILES ${SRC_FILES} PARENT_SCOPE)