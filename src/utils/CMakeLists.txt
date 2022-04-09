set(utils_source_files
        SignPadding.h
        SignPadding.cpp
        CSV.h
        CSV.cpp
        )

add_library(utils ${utils_source_files})

target_link_libraries(utils ode)