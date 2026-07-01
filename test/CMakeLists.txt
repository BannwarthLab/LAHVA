# Main test CMakeLists.txt
# Mirrors the project structure: test/cpu and test/gpu

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

# Coverage target
if(ENABLE_COVERAGE)
    find_program(GCOV gcov)
    find_program(LCOV lcov)

    if(LCOV)
        add_custom_target(coverage
            COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
            COMMAND ${LCOV} --capture --directory . --output-file coverage.info
            COMMAND ${LCOV} --remove coverage.info '/usr/*' '${PROJECT_SOURCE_DIR}/include/netlib/*' --output-file coverage.info
            COMMAND ${LCOV} --list coverage.info
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Generating coverage report"
        )
    else()
        message(WARNING "lcov not found, coverage target will not be available")
    endif()
endif()
