file(GLOB_RECURSE qm_sources "*.cu")
file(GLOB_RECURSE qm_headers "${PROJECT_SOURCE_DIR}/include/*.h" "${PROJECT_SOURCE_DIR}/include/**.cuh")

add_library(${qm_lib} ${qm_sources} ${qm_headers})

target_link_libraries(${qm_lib} ${la_lib})

#include all headers in the include directory
target_include_directories(${qm_lib} PUBLIC "${PROJECT_SOURCE_DIR}/include")