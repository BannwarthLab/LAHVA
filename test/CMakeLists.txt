add_executable(${PROJECT_NAME}-level1-tester blas/level1.cpp utils.cpp)
target_link_libraries(${PROJECT_NAME}-level1-tester PRIVATE ${PROJECT_NAME})
add_test(NAME level1 COMMAND ${PROJECT_NAME}-level1-tester)

add_executable(${PROJECT_NAME}-level2-tester blas/level2.cpp utils.cpp)
target_link_libraries(${PROJECT_NAME}-level2-tester PRIVATE ${PROJECT_NAME})
add_test(NAME level2 COMMAND ${PROJECT_NAME}-level2-tester)

add_executable(${PROJECT_NAME}-level3-tester blas/level3.cpp utils.cpp)
target_link_libraries(${PROJECT_NAME}-level3-tester PRIVATE ${PROJECT_NAME})
add_test(NAME level3 COMMAND ${PROJECT_NAME}-level3-tester)

add_executable(${PROJECT_NAME}-add-level2-tester blas/additional-level2.cpp utils.cpp)
target_link_libraries(${PROJECT_NAME}-add-level2-tester PRIVATE ${PROJECT_NAME})
add_test(NAME additional-level-2 COMMAND ${PROJECT_NAME}-add-level2-tester)

add_executable(${PROJECT_NAME}-add-level3-tester blas/additional-level3.cpp utils.cpp)
target_link_libraries(${PROJECT_NAME}-add-level3-tester PRIVATE ${PROJECT_NAME})
add_test(NAME additional-level-3 COMMAND ${PROJECT_NAME}-add-level3-tester)

add_executable(${PROJECT_NAME}-blockdiagmatrix-tester tensor/blockdiagmatrix.cpp)
target_link_libraries(${PROJECT_NAME}-blockdiagmatrix-tester PRIVATE ${PROJECT_NAME})
add_test(NAME blockdiagmatrix COMMAND ${PROJECT_NAME}-blockdiagmatrix-tester)

# GPU tests
if(GPU)
  add_executable(${PROJECT_NAME}-level1-gpu-tester blas/level1-gpu.cpp utils.cpp)
  target_link_libraries(${PROJECT_NAME}-level1-gpu-tester PRIVATE ${PROJECT_NAME})
  add_test(NAME level1-gpu COMMAND ${PROJECT_NAME}-level1-gpu-tester)

  add_executable(${PROJECT_NAME}-level2-gpu-tester blas/level2-gpu.cpp utils.cpp)
  target_link_libraries(${PROJECT_NAME}-level2-gpu-tester PRIVATE ${PROJECT_NAME})
  add_test(NAME level2-gpu COMMAND ${PROJECT_NAME}-level2-gpu-tester)

  add_executable(${PROJECT_NAME}-add-level1-gpu-tester blas/additional-level1-gpu.cpp utils.cpp)
  target_link_libraries(${PROJECT_NAME}-add-level1-gpu-tester PRIVATE ${PROJECT_NAME})
  add_test(NAME additional-level-1-gpu COMMAND ${PROJECT_NAME}-add-level1-gpu-tester)

  add_executable(${PROJECT_NAME}-add-level2-gpu-tester blas/additional-level2-gpu.cpp utils.cpp)
  target_link_libraries(${PROJECT_NAME}-add-level2-gpu-tester PRIVATE ${PROJECT_NAME})
  add_test(NAME additional-level-2-gpu COMMAND ${PROJECT_NAME}-add-level2-gpu-tester)

  add_executable(${PROJECT_NAME}-add-level3-gpu-tester blas/additional-level3-gpu.cpp utils.cpp)
  target_link_libraries(${PROJECT_NAME}-add-level3-gpu-tester PRIVATE ${PROJECT_NAME})
  add_test(NAME additional-level-3-gpu COMMAND ${PROJECT_NAME}-add-level3-gpu-tester)
  
  add_executable(${PROJECT_NAME}-gpu-misc tensor/gpu-misc.cpp)
  target_link_libraries(${PROJECT_NAME}-gpu-misc PRIVATE ${PROJECT_NAME})
  add_test(NAME gpu-misc COMMAND ${PROJECT_NAME}-gpu-misc)

  add_executable(${PROJECT_NAME}-mat-attributes-tester tensor/mat-attributes.cpp utils.cpp)
  target_link_libraries(${PROJECT_NAME}-mat-attributes-tester PRIVATE ${PROJECT_NAME})
  add_test(NAME mat-attributes COMMAND ${PROJECT_NAME}-mat-attributes-tester)
  
  add_executable(${PROJECT_NAME}-test-mp tensor/test-mp.cpp)
  target_link_libraries(${PROJECT_NAME}-test-mp PRIVATE ${PROJECT_NAME})
  add_test(NAME test-mp COMMAND ${PROJECT_NAME}-test-mp)

endif()