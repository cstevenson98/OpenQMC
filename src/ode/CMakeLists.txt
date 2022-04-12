set(ode_source_files
        Integrator.cpp
        Integrator.h
        Euler.cpp
        Euler.h
        Euler_GPU.cu
        Euler_GPU.cuh
        RK4.cpp
        RK4.h
        GPU.cu
        GPU.cuh
        )

add_library(ode ${ode_source_files})

target_link_libraries(ode la)
set_target_properties(ode PROPERTIES CUDA_ARCHITECTURES "75")
