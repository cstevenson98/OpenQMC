set(qm_source_files
        Spins.cpp
        Spins.h
        )
add_library(qm ${qm_source_files})

target_link_libraries(qm la)
