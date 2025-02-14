
# Level 1 test
add_executable(${PROJECT_NAME}-level1-tester level1.cpp utils.cpp)
target_link_libraries(${PROJECT_NAME}-level1-tester PRIVATE ${PROJECT_NAME})
add_test(NAME level1 COMMAND ${PROJECT_NAME}-level1-tester)

# Level 2 test
add_executable(${PROJECT_NAME}-level2-tester level2.cpp utils.cpp)
target_link_libraries(${PROJECT_NAME}-level2-tester PRIVATE ${PROJECT_NAME})
add_test(NAME level2 COMMAND ${PROJECT_NAME}-level2-tester)

# Level 3 test
add_executable(${PROJECT_NAME}-level3-tester level3.cpp utils.cpp)
target_link_libraries(${PROJECT_NAME}-level3-tester PRIVATE ${PROJECT_NAME})
add_test(NAME level3 COMMAND ${PROJECT_NAME}-level3-tester)

# Additional Level 2 test
add_executable(${PROJECT_NAME}-add-level2-tester additional-level2.cpp utils.cpp)
target_link_libraries(${PROJECT_NAME}-add-level2-tester PRIVATE ${PROJECT_NAME})
add_test(NAME additional-level-2 COMMAND ${PROJECT_NAME}-add-level2-tester)

# GPU tests
if(GPU)
  add_executable(${PROJECT_NAME}-level1-gpu-tester level1-gpu.cpp utils.cpp)
  target_link_libraries(${PROJECT_NAME}-level1-gpu-tester PRIVATE ${PROJECT_NAME})
  add_test(NAME level1-gpu COMMAND ${PROJECT_NAME}-level1-gpu-tester)

  add_executable(${PROJECT_NAME}-level2-gpu-tester level2-gpu.cpp utils.cpp)
  target_link_libraries(${PROJECT_NAME}-level2-gpu-tester PRIVATE ${PROJECT_NAME})
  add_test(NAME level2-gpu COMMAND ${PROJECT_NAME}-level2-gpu-tester)

  add_executable(${PROJECT_NAME}-add-level2-gpu-tester additional-level2-gpu.cpp utils.cpp)
  target_link_libraries(${PROJECT_NAME}-add-level2-gpu-tester PRIVATE ${PROJECT_NAME})
  add_test(NAME additional-level-2-gpu COMMAND ${PROJECT_NAME}-add-level2-gpu-tester)

endif()