# CPU Tests - mirrors src/cpu structure
# Tests for CPU implementation (both C and C++ API)

# Type/Class Tests
# Vector type tests
add_executable(${PROJECT_NAME}-cpu-vector-types-test
    types/vector-types.cpp
    $<TARGET_OBJECTS:test-utils>
)
target_link_libraries(${PROJECT_NAME}-cpu-vector-types-test PRIVATE ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}-cpu-vector-types-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
add_test(NAME cpu-types-vector COMMAND ${PROJECT_NAME}-cpu-vector-types-test)

# Matrix type tests
add_executable(${PROJECT_NAME}-cpu-matrix-types-test
    types/matrix-types.cpp
    $<TARGET_OBJECTS:test-utils>
)
target_link_libraries(${PROJECT_NAME}-cpu-matrix-types-test PRIVATE ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}-cpu-matrix-types-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
add_test(NAME cpu-types-matrix COMMAND ${PROJECT_NAME}-cpu-matrix-types-test)

# LowerTriMatrix type tests
add_executable(${PROJECT_NAME}-cpu-lowtrimatrix-types-test
    types/lowtrimatrix-types.cpp
    $<TARGET_OBJECTS:test-utils>
)
target_link_libraries(${PROJECT_NAME}-cpu-lowtrimatrix-types-test PRIVATE ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}-cpu-lowtrimatrix-types-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
add_test(NAME cpu-types-lowtrimatrix COMMAND ${PROJECT_NAME}-cpu-lowtrimatrix-types-test)

# BlockDiagMatrix type tests
add_executable(${PROJECT_NAME}-cpu-blockdiagmatrix-types-test
    types/blockdiagmatrix-types.cpp
    $<TARGET_OBJECTS:test-utils>
)
target_link_libraries(${PROJECT_NAME}-cpu-blockdiagmatrix-types-test PRIVATE ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}-cpu-blockdiagmatrix-types-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
add_test(NAME cpu-types-blockdiagmatrix COMMAND ${PROJECT_NAME}-cpu-blockdiagmatrix-types-test)

# Level 1 BLAS tests (impl)
add_executable(${PROJECT_NAME}-cpu-level1-test
    impl/level1.cpp
    $<TARGET_OBJECTS:test-utils>
)
target_link_libraries(${PROJECT_NAME}-cpu-level1-test PRIVATE ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}-cpu-level1-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
add_test(NAME cpu-blas-level1 COMMAND ${PROJECT_NAME}-cpu-level1-test)

# Level 2 BLAS tests (impl)
add_executable(${PROJECT_NAME}-cpu-level2-test
    impl/level2.cpp
    $<TARGET_OBJECTS:test-utils>
)
target_link_libraries(${PROJECT_NAME}-cpu-level2-test PRIVATE ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}-cpu-level2-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
add_test(NAME cpu-blas-level2 COMMAND ${PROJECT_NAME}-cpu-level2-test)

# Level 3 BLAS tests (impl)
add_executable(${PROJECT_NAME}-cpu-level3-test
    impl/level3.cpp
    $<TARGET_OBJECTS:test-utils>
)
target_link_libraries(${PROJECT_NAME}-cpu-level3-test PRIVATE ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}-cpu-level3-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
add_test(NAME cpu-blas-level3 COMMAND ${PROJECT_NAME}-cpu-level3-test)

# Additional Level 1 tests (additional-blas)
add_executable(${PROJECT_NAME}-cpu-additional-level1-test
    blas/additional/level1.cpp
    $<TARGET_OBJECTS:test-utils>
)
target_link_libraries(${PROJECT_NAME}-cpu-additional-level1-test PRIVATE ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}-cpu-additional-level1-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
add_test(NAME cpu-additional-level1 COMMAND ${PROJECT_NAME}-cpu-additional-level1-test)

# LAPACK tests
add_executable(${PROJECT_NAME}-cpu-lapack-test
    lapack-test.cpp
    $<TARGET_OBJECTS:test-utils>
)
target_link_libraries(${PROJECT_NAME}-cpu-lapack-test PRIVATE ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}-cpu-lapack-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
add_test(NAME cpu-lapack COMMAND ${PROJECT_NAME}-cpu-lapack-test)