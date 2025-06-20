add_subdirectory(cpu)

if(GPU)
  add_subdirectory(gpu)
endif()

set(SRC_FILES ${SRC_FILES} PARENT_SCOPE)
set(HEADER_FILES ${HEADER_FILES} PARENT_SCOPE)