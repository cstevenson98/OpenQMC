cmake_minimum_required(VERSION 3.31 FATAL_ERROR)
# CUDA SETUP

project(OpenQMC LANGUAGES CXX CUDA)

# STD
set(CMAKE_CXX_STANDARD 23)

add_subdirectory(src)

set(main_files main.cpp)
add_executable(main ${main_files})
target_link_libraries(main oqmc)
