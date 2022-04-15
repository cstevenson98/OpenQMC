cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(main LANGUAGES CUDA CXX)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set (CMAKE_CUDA_FLAGS "${CMAKE_CXX_FLAGS} -G")
# BOOST SETUP
# find_package(Boost 1.71.0 COMPONENTS filesystem)
# set(Boost_USE_STATIC_LIBS OFF)
# set(Boost_USE_MULTITHREADED ON)
# set(Boost_USE_STATIC_RUNTIME OFF)
# set(Boost_DEBUG ON)

# CUDA SETUP
find_package(CUDA REQUIRED)
set (CMAKE_CUDA_COMPILER /usr/local/cuda-11.5/bin/nvcc)
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# OPENCL, REQUIRED BY BOOST::COMPUTE
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})

# LOCAL PACKAGES
add_subdirectory(utils)
add_subdirectory(la)
add_subdirectory(qm)
add_subdirectory(ode)
add_subdirectory(models)

# MAIN
set(main_files main.cu)
add_executable(main ${main_files})
set_target_properties(main PROPERTIES CUDA_ARCHITECTURES "75")
target_link_libraries(main utils la qm models ode)
target_include_directories(main PRIVATE /home/conor/dev/thrust/cmake/)

# ELASTIC
set(elastic_files elastic.cu)
add_executable(elastic ${elastic_files})
set_target_properties(main PROPERTIES CUDA_ARCHITECTURES "75")
target_link_libraries(elastic utils la qm models ode)

