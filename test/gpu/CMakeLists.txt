# GPU Tests - mirrors src/gpu structure
# Tests for GPU BLAS and additional-BLAS operations
# Requires CUDA to be enabled

# Type/Class Tests
# GPU Vector type tests
add_executable(${PROJECT_NAME}-gpu-vector-types-test
    types/vector-types.cpp
    $<TARGET_OBJECTS:test-utils>
)
target_link_libraries(${PROJECT_NAME}-gpu-vector-types-test PRIVATE ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}-gpu-vector-types-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
add_test(NAME gpu-types-vector COMMAND ${PROJECT_NAME}-gpu-vector-types-test)

# GPU Matrix type tests
add_executable(${PROJECT_NAME}-gpu-matrix-types-test
    types/matrix-types.cpp
    $<TARGET_OBJECTS:test-utils>
)
target_link_libraries(${PROJECT_NAME}-gpu-matrix-types-test PRIVATE ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}-gpu-matrix-types-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
add_test(NAME gpu-types-matrix COMMAND ${PROJECT_NAME}-gpu-matrix-types-test)

# GPU LowTriMatrix type tests
add_executable(${PROJECT_NAME}-gpu-lowtrimatrix-types-test
    types/lowtrimatrix-types.cpp
    $<TARGET_OBJECTS:test-utils>
)
target_link_libraries(${PROJECT_NAME}-gpu-lowtrimatrix-types-test PRIVATE ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}-gpu-lowtrimatrix-types-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
add_test(NAME gpu-types-lowtrimatrix COMMAND ${PROJECT_NAME}-gpu-lowtrimatrix-types-test)

# GPU miscellaneous tests
add_executable(${PROJECT_NAME}-gpu-misc-test
    utils/gpu-misc.cpp
    $<TARGET_OBJECTS:test-utils>
)
target_link_libraries(${PROJECT_NAME}-gpu-misc-test PRIVATE ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}-gpu-misc-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
add_test(NAME gpu-misc COMMAND ${PROJECT_NAME}-gpu-misc-test)

# Level 1 BLAS tests (blas)
add_executable(${PROJECT_NAME}-gpu-level1-test
    blas/level1.cpp
    $<TARGET_OBJECTS:test-utils>
)
target_link_libraries(${PROJECT_NAME}-gpu-level1-test PRIVATE ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}-gpu-level1-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
add_test(NAME gpu-blas-level1 COMMAND ${PROJECT_NAME}-gpu-level1-test)

# Level 2 BLAS tests (blas)
add_executable(${PROJECT_NAME}-gpu-level2-test
    blas/level2.cpp
    $<TARGET_OBJECTS:test-utils>
)
target_link_libraries(${PROJECT_NAME}-gpu-level2-test PRIVATE ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}-gpu-level2-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
add_test(NAME gpu-blas-level2 COMMAND ${PROJECT_NAME}-gpu-level2-test)

# Level 3 BLAS tests (blas)
add_executable(${PROJECT_NAME}-gpu-level3-test
    blas/level3.cpp
    $<TARGET_OBJECTS:test-utils>
)
target_link_libraries(${PROJECT_NAME}-gpu-level3-test PRIVATE ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}-gpu-level3-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
add_test(NAME gpu-blas-level3 COMMAND ${PROJECT_NAME}-gpu-level3-test)

# Additional Level 1 tests (additional-blas)
add_executable(${PROJECT_NAME}-gpu-additional-level1-test
    additional/level1.cpp
    $<TARGET_OBJECTS:test-utils>
)
target_link_libraries(${PROJECT_NAME}-gpu-additional-level1-test PRIVATE ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}-gpu-additional-level1-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
add_test(NAME gpu-additional-level1 COMMAND ${PROJECT_NAME}-gpu-additional-level1-test)

# Additional Level 2 tests (additional-blas)
add_executable(${PROJECT_NAME}-gpu-additional-level2-test
    additional/level2.cpp
    $<TARGET_OBJECTS:test-utils>
)
target_link_libraries(${PROJECT_NAME}-gpu-additional-level2-test PRIVATE ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}-gpu-additional-level2-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
add_test(NAME gpu-additional-level2 COMMAND ${PROJECT_NAME}-gpu-additional-level2-test)

# Mixed precision tests
add_executable(${PROJECT_NAME}-gpu-test-mp
    utils/test-mp.cpp
    $<TARGET_OBJECTS:test-utils>
)
target_link_libraries(${PROJECT_NAME}-gpu-test-mp PRIVATE ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}-gpu-test-mp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
add_test(NAME gpu-mixed-precision COMMAND ${PROJECT_NAME}-gpu-test-mp)

