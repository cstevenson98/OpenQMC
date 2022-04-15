set(utils_source_files
        SignPadding.cuh
        SignPadding.cu
        CSV.cuh
        CSV.cu
        )

add_library(utils ${utils_source_files})

target_link_libraries(utils ode)