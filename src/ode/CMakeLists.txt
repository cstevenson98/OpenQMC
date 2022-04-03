set(ode_source_files integrator.cpp integrator.h)
add_library(ode ${ode_source_files})

target_link_libraries(la)
