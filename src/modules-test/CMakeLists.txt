# Create a library for the module interface
add_executable(math_module main.cc)
target_sources(math_module
    PUBLIC
    FILE_SET all_my_modules TYPE CXX_MODULES FILES
    math.ixx
)
