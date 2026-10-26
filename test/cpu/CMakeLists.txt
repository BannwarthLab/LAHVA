# Tensor type tests
# Vector tensor tests
add_executable(${PROJECT_NAME}-cpu-tensor-vector
    tensor/vector.cpp
)
target_link_libraries(${PROJECT_NAME}-cpu-tensor-vector PRIVATE ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}-cpu-tensor-vector PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
add_test(NAME cpu-tensor-vector COMMAND ${PROJECT_NAME}-cpu-tensor-vector)

# Matrix tensor tests
add_executable(${PROJECT_NAME}-cpu-tensor-matrix
    tensor/matrix.cpp
)
target_link_libraries(${PROJECT_NAME}-cpu-tensor-matrix PRIVATE ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}-cpu-tensor-matrix PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
add_test(NAME cpu-tensor-matrix COMMAND ${PROJECT_NAME}-cpu-tensor-matrix)

# LowerTriMatrix tensor tests
add_executable(${PROJECT_NAME}-cpu-tensor-lowtrimatrix
    tensor/lowtrimatrix.cpp
)
target_link_libraries(${PROJECT_NAME}-cpu-tensor-lowtrimatrix PRIVATE ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}-cpu-tensor-lowtrimatrix PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
add_test(NAME cpu-tensor-lowtrimatrix COMMAND ${PROJECT_NAME}-cpu-tensor-lowtrimatrix)

# Level 1 BLAS tests
add_executable(${PROJECT_NAME}-cpu-level1
    blas/level1.cpp
)
target_link_libraries(${PROJECT_NAME}-cpu-level1 PRIVATE ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}-cpu-level1 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
add_test(NAME cpu-blas-level1 COMMAND ${PROJECT_NAME}-cpu-level1)

# Level 2 BLAS tests
add_executable(${PROJECT_NAME}-cpu-level2
    blas/level2.cpp
)
target_link_libraries(${PROJECT_NAME}-cpu-level2 PRIVATE ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}-cpu-level2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
add_test(NAME cpu-blas-level2 COMMAND ${PROJECT_NAME}-cpu-level2)

# Level 3 BLAS tests
add_executable(${PROJECT_NAME}-cpu-level3
    blas/level3.cpp
)
target_link_libraries(${PROJECT_NAME}-cpu-level3 PRIVATE ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}-cpu-level3 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
add_test(NAME cpu-blas-level3 COMMAND ${PROJECT_NAME}-cpu-level3)

# Additional Level 1 tests
add_executable(${PROJECT_NAME}-cpu-blas-additional-level1
    blas/additional-level1.cpp
)
target_link_libraries(${PROJECT_NAME}-cpu-blas-additional-level1 PRIVATE ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}-cpu-blas-additional-level1 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
add_test(NAME cpu-blas-additional-level1 COMMAND ${PROJECT_NAME}-cpu-blas-additional-level1)

# Additional Level 2 tests
add_executable(${PROJECT_NAME}-cpu-blas-additional-level2
    blas/additional-level2.cpp
)
target_link_libraries(${PROJECT_NAME}-cpu-blas-additional-level2 PRIVATE ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}-cpu-blas-additional-level2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
add_test(NAME cpu-blas-additional-level2 COMMAND ${PROJECT_NAME}-cpu-blas-additional-level2)

# LAPACK tests
add_executable(${PROJECT_NAME}-cpu-blas-lapack
    blas/lapack.cpp
)
target_link_libraries(${PROJECT_NAME}-cpu-blas-lapack PRIVATE ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}-cpu-blas-lapack PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
add_test(NAME cpu-blas-lapack COMMAND ${PROJECT_NAME}-cpu-blas-lapack)