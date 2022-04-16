set(ode_source_files
        Integrator.cu
        Integrator.cuh
        Euler.cu
        Euler.cuh
        Euler_GPU.cu
        Euler_GPU.cuh
        RK4.cu
        RK4.cuh
        RK4_GPU.cu
        RK4_GPU.cuh
        GPU.cu
        GPU.cuh
        )

add_library(ode ${ode_source_files})

target_link_libraries(ode la)
set_target_properties(ode PROPERTIES CUDA_ARCHITECTURES "75")
