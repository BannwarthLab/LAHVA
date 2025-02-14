add_subdirectory(cpu)
add_subdirectory(utils)

if(GPU)
  add_subdirectory(gpu)
  add_subdirectory(gpu-utils)
endif()

set(SRC_FILES ${SRC_FILES} PARENT_SCOPE)
set(HEADER_FILES ${HEADER_FILES} PARENT_SCOPE)