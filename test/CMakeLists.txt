enable_testing()

# Coverage options
option(ENABLE_COVERAGE "Enable code coverage" OFF)
if(ENABLE_COVERAGE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -fprofile-arcs -ftest-coverage")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage -fprofile-arcs -ftest-coverage")
endif()

# Common test utilities
add_library(test-utils OBJECT
    common/utils.cpp
)
target_include_directories(test-utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${PROJECT_SOURCE_DIR}/include
)

# Add CPU tests
add_subdirectory(cpu)

# Add GPU tests if CUDA is enabled
if(GPU)
    add_subdirectory(gpu)
endif()