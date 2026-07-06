# GPU Tests

# GPU Tensor type tests
# GPU Tensor tests
add_executable(${PROJECT_NAME}-gpu-tensor-tensor
    tensor/gpu-tensor.cpp
    $<TARGET_OBJECTS:test-utils>
)
target_link_libraries(${PROJECT_NAME}-gpu-tensor-tensor PRIVATE ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}-gpu-tensor-tensor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
add_test(NAME gpu-tensor-tensor COMMAND ${PROJECT_NAME}-gpu-tensor-tensor)

# Vector tensor tests
add_executable(${PROJECT_NAME}-gpu-vector-tensor
    tensor/vector.cpp
    $<TARGET_OBJECTS:test-utils>
)
target_link_libraries(${PROJECT_NAME}-gpu-vector-tensor PRIVATE ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}-gpu-vector-tensor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
add_test(NAME gpu-tensor-vector COMMAND ${PROJECT_NAME}-gpu-vector-tensor)

# Matrix tensor tests
add_executable(${PROJECT_NAME}-gpu-matrix-tensor
    tensor/matrix.cpp
    $<TARGET_OBJECTS:test-utils>
)
target_link_libraries(${PROJECT_NAME}-gpu-matrix-tensor PRIVATE ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}-gpu-matrix-tensor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
add_test(NAME gpu-tensor-matrix COMMAND ${PROJECT_NAME}-gpu-matrix-tensor)

# LowTriMatrix tensor tests
add_executable(${PROJECT_NAME}-gpu-lowtrimatrix-tensor
    tensor/lowtrimatrix.cpp
    $<TARGET_OBJECTS:test-utils>
)
target_link_libraries(${PROJECT_NAME}-gpu-lowtrimatrix-tensor PRIVATE ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}-gpu-lowtrimatrix-tensor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
add_test(NAME gpu-tensor-lowtrimatrix COMMAND ${PROJECT_NAME}-gpu-lowtrimatrix-tensor)

# Mixed-Precision Matrix tensor tests
add_executable(${PROJECT_NAME}-gpu-tensor-mixed-precision-matrix
    tensor/mixed-precision-matrix.cpp
    $<TARGET_OBJECTS:test-utils>
)
target_link_libraries(${PROJECT_NAME}-gpu-tensor-mixed-precision-matrix PRIVATE ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}-gpu-tensor-mixed-precision-matrix PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
add_test(NAME gpu-tensor-mixed-precision-matrix COMMAND ${PROJECT_NAME}-gpu-tensor-mixed-precision-matrix)

# GPU miscellaneous tests
add_executable(${PROJECT_NAME}-gpu-utils-misc
    utils/gpu-misc.cpp
    $<TARGET_OBJECTS:test-utils>
)
target_link_libraries(${PROJECT_NAME}-gpu-utils-misc PRIVATE ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}-gpu-utils-misc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
add_test(NAME gpu-utils-misc COMMAND ${PROJECT_NAME}-gpu-utils-misc)

# Level 1 BLAS tests
add_executable(${PROJECT_NAME}-gpu-level1
    blas/level1.cpp
    $<TARGET_OBJECTS:test-utils>
)
target_link_libraries(${PROJECT_NAME}-gpu-level1 PRIVATE ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}-gpu-level1 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
add_test(NAME gpu-blas-level1 COMMAND ${PROJECT_NAME}-gpu-level1)

# Level 2 BLAS tests
add_executable(${PROJECT_NAME}-gpu-level2
    blas/level2.cpp
    $<TARGET_OBJECTS:test-utils>
)
target_link_libraries(${PROJECT_NAME}-gpu-level2 PRIVATE ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}-gpu-level2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
add_test(NAME gpu-blas-level2 COMMAND ${PROJECT_NAME}-gpu-level2)

# Level 3 BLAS tests
add_executable(${PROJECT_NAME}-gpu-level3
    blas/level3.cpp
    $<TARGET_OBJECTS:test-utils>
)
target_link_libraries(${PROJECT_NAME}-gpu-level3 PRIVATE ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}-gpu-level3 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
add_test(NAME gpu-blas-level3 COMMAND ${PROJECT_NAME}-gpu-level3)

# Additional Level 1 tests
add_executable(${PROJECT_NAME}-gpu-blas-additional-level1
    blas/additional-level1.cpp
    $<TARGET_OBJECTS:test-utils>
)
target_link_libraries(${PROJECT_NAME}-gpu-blas-additional-level1 PRIVATE ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}-gpu-blas-additional-level1 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
add_test(NAME gpu-blas-additional-level1 COMMAND ${PROJECT_NAME}-gpu-blas-additional-level1)

# Additional Level 2 tests
add_executable(${PROJECT_NAME}-gpu-additional-level2
    blas/additional-level2.cpp
    $<TARGET_OBJECTS:test-utils>
)
target_link_libraries(${PROJECT_NAME}-gpu-additional-level2 PRIVATE ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}-gpu-additional-level2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
add_test(NAME gpu-blas-additional-level2 COMMAND ${PROJECT_NAME}-gpu-additional-level2)

# GPU mixed precision tests
add_executable(${PROJECT_NAME}-gpu-test-mp
    utils/test-mp.cpp
    $<TARGET_OBJECTS:test-utils>
)
target_link_libraries(${PROJECT_NAME}-gpu-test-mp PRIVATE ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}-gpu-test-mp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
add_test(NAME gpu-mixed-precision COMMAND ${PROJECT_NAME}-gpu-test-mp)
