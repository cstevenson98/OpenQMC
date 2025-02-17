# glob all source files
file(GLOB_RECURSE models_sources "*.cu")
file(GLOB_RECURSE models_headers "${PROJECT_SOURCE_DIR}/include/*.h" "${PROJECT_SOURCE_DIR}/include/**.cuh")

add_library(${models_lib} ${models_sources} ${models_headers})

target_link_libraries(${models_lib} ${qm_lib})

#include all headers in the include directory
target_include_directories(${models_lib} PUBLIC "${PROJECT_SOURCE_DIR}/include")