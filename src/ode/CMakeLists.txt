set(ode_source_files
        Integrator.cpp
        Integrator.h
        Euler.cpp
        Euler.h
        RK4.cpp
        RK4.h
        )

add_library(ode ${ode_source_files})

target_link_libraries(ode la)
