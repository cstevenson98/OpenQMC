set(ode_lib "ode")
set(models_lib "models")
set(qm_lib "qm")
set(la_lib "la")
set(utils_lib "utils")

find_package(Eigen3 REQUIRED)

# Enable CUDA and find cuSPARSE
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# MAIN
file(GLOB_RECURSE oqmc_sources "${PROJECT_SOURCE_DIR}/src/*.cu" "${PROJECT_SOURCE_DIR}/src/*.cuh")
file(GLOB_RECURSE oqmc_headers "${PROJECT_SOURCE_DIR}/include/*.h" "${PROJECT_SOURCE_DIR}/include/**.cuh")

# add_library(oqmc ${oqmc_sources} ${oqmc_headers})
add_library(oqmc ${oqmc_sources} ${oqmc_headers})
target_include_directories(oqmc PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(oqmc PUBLIC Eigen3::Eigen CUDA::cusparse)

# # ELASTIC
# set(elastic_files elastic.cu)
# add_executable(elastic ${elastic_files})
# target_link_libraries(elastic ${utils_lib} ${la_lib} ${qm_lib} ${models_lib} ${ode_lib})