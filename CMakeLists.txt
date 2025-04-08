cmake_minimum_required(VERSION 3.31 FATAL_ERROR)
# CUDA SETUP

project(OpenQMC LANGUAGES CXX CUDA)

# STD
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable C++20 modules support if needed
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    # For GCC
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fmodules-ts")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # For Clang - use the correct flag
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fmodules")
elseif(MSVC)
    # For MSVC
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /experimental:module")
endif()

# Set module output directory
set(CMAKE_CXX_MODULE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/modules)

add_subdirectory(src)

set(main_files main.cpp)
add_executable(main ${main_files})
target_link_libraries(main oqmc)

# testing directory
add_subdirectory(test)
