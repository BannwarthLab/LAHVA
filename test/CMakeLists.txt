enable_testing()

add_subdirectory(cpu)

if(GPU)
    add_subdirectory(gpu)
endif()