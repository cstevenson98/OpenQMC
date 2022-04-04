cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

project(main)

add_subdirectory(la)
add_subdirectory(ode)
add_subdirectory(models)

set(source_files main.cpp)
add_executable(main ${source_files})

target_link_libraries(main la ode models)
