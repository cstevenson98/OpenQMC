# glob all source files
file(GLOB_RECURSE la_files "*.cpp" "*.cu")
file(GLOB_RECURSE la_headers "${PROJECT_SOURCE_DIR}/include/la/*.h" "${PROJECT_SOURCE_DIR}/include/la/*.cuh")

# add library
add_library(la ${la_files} ${la_headers})
# link libraries
target_link_libraries(la)
# include directories
target_include_directories(la PUBLIC ${PROJECT_SOURCE_DIR}/include/la)
