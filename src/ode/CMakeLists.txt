set(ode_source_files
        Integrator.cpp
        Integrator.h
        Euler.cpp
        Euler.h
        )

add_library(ode ${ode_source_files})

target_link_libraries(la)
