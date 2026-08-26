add_executable(mulliken main.cpp mulliken.cpp)
target_link_libraries(mulliken PRIVATE "${PROJECT_NAME}-lib")
target_include_directories(mulliken PRIVATE ..)
