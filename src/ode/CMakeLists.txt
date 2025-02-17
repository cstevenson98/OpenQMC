# glob all source files
file(GLOB_RECURSE ode_sources "*.cu")
file(GLOB_RECURSE ode_headers "${PROJECT_SOURCE_DIR}/include/*.h" "${PROJECT_SOURCE_DIR}/include/**.cuh")

add_library(${ode_lib} ${ode_sources} ${ode_headers})

target_link_libraries(${ode_lib} ${la_lib})

#include all headers in the include directory
target_include_directories(${ode_lib} PUBLIC "${PROJECT_SOURCE_DIR}/include")