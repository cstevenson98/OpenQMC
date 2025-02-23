# glob all source files
file(GLOB_RECURSE la_sources "Dense.cu")
file(GLOB_RECURSE la_headers "${PROJECT_SOURCE_DIR}/include/*.h" "${PROJECT_SOURCE_DIR}/include/**.cuh")

add_library(${la_lib} ${la_sources} ${la_headers})

target_link_libraries(${la_lib} ${utils_lib})
set_property(TARGET ${la_lib} PROPERTY CUDA_ARCHITECTURES native)

#include all headers in the include directory
target_include_directories(${la_lib} PUBLIC "${PROJECT_SOURCE_DIR}/include")