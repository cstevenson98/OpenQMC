# BOOST SETUP
find_package(Boost 1.71.0 COMPONENTS filesystem)
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
set(Boost_DEBUG ON)

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

# TESTS
# add_subdirectory(Boost_tests)

# MAIN
set(main_files main.cu)
add_executable(main ${main_files})
# set_target_properties(main PROPERTIES CUDA_ARCHITECTURES "75")
target_link_libraries(main utils la qm models ode)
target_include_directories(main PRIVATE /home/conor/dev/thrust/cmake/)

# ELASTIC
set(elastic_files elastic.cu)
add_executable(elastic ${elastic_files})
# set_target_properties(main PROPERTIES CUDA_ARCHITECTURES "75")
target_link_libraries(elastic utils la qm models ode)

