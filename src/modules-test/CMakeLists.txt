

set(library_files
    test.cu
)
add_library(cu_test SHARED ${library_files})
include_directories(include)

# Create a library for the module interface
add_executable(math_module main.cc)
target_sources(math_module
    PUBLIC
    FILE_SET all_my_modules TYPE CXX_MODULES FILES
    math.ixx
    math-complex.ixx
)

target_link_libraries(math_module PRIVATE cu_test)