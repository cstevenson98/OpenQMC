set(qm_source_files
        Spins.cu
        Spins.cuh
        Open.cu
        Open.cuh
        )

add_library(qm ${qm_source_files})

target_link_libraries(qm la)
