cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

project(main)

add_subdirectory(utils)
add_subdirectory(la)
add_subdirectory(qm)
add_subdirectory(ode)
add_subdirectory(models)

set(main_files main.cpp)
add_executable(main ${main_files})

set(elastic_files elastic.cpp)
add_executable(elastic ${elastic_files})

target_link_libraries(main utils la qm models ode)
target_link_libraries(elastic utils la qm models ode)
