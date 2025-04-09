cmake_minimum_required(VERSION 3.31 FATAL_ERROR)
# CUDA SETUP
# Enables the Standard module support. This needs to be done
# before selecting the languages.
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "0e5b6991-d74f-4b3d-a41c-cf096e0b2508")
set(CMAKE_CXX_MODULE_STD ON)
# STD
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
project(OpenQMC LANGUAGES CXX)



# # Enable C++20 modules support if needed
# if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
#     # For GCC
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fmodules-ts")
# elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#     # For Clang - use the correct flag
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fmodules")
# elseif(MSVC)
#     # For MSVC
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /experimental:module")
# endif()

# Set module output directory
set(CMAKE_CXX_MODULE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/modules)

add_subdirectory(src)

# add_executable(main main.cpp) 
# include 


# target_link_libraries(main oqmc)

# # testing directory
# add_subdirectory(test)
