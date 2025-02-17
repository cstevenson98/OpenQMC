# BOOST SETUP
# find_package(Boost 1.71.0 COMPONENTS filesystem)
# set(Boost_USE_STATIC_LIBS OFF)
# set(Boost_USE_MULTITHREADED ON)
# set(Boost_USE_STATIC_RUNTIME OFF)
# set(Boost_DEBUG ON)

# OPENCL, REQUIRED BY BOOST::COMPUTE
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})


set(ode_lib "ode")
set(models_lib "models")
set(qm_lib "qm")
set(la_lib "la")
set(utils_lib "utils")

# LOCAL PACKAGES
add_subdirectory(utils)
add_subdirectory(la)
add_subdirectory(qm)
add_subdirectory(ode)
add_subdirectory(models)

# MAIN
set(main_files main.cu)
add_executable(main ${main_files})
target_link_libraries(main ${utils_lib} ${la_lib} ${qm_lib} ${models_lib} ${ode_lib})
# target_include_directories(main PRIVATE /home/conor/dev/thrust/cmake/)

# ELASTIC
# set(elastic_files elastic.cu)
# add_executable(elastic ${elastic_files})
# target_link_libraries(elastic utils la qm models ode)

