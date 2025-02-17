# glob all source files
file(GLOB_RECURSE utils_sources "*.cu")
file(GLOB_RECURSE utils_headers "${PROJECT_SOURCE_DIR}/include/*.h" "${PROJECT_SOURCE_DIR}/include/**.cuh")

add_library(${utils_lib} ${utils_sources} ${utils_headers})

target_link_libraries(${utils_lib} ${ode_lib})

#include all headers in the include directory
target_include_directories(${utils_lib} PUBLIC "${PROJECT_SOURCE_DIR}/include")