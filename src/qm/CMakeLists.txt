set(qm_source_files
        Spins.cpp
        Spins.h
        ../models/ElasticChain1D.cpp
        ../models/ElasticChain1D.h
        ../models/XYZModel.cpp
        ../models/XYZModel.h
        )

add_library(qm ${qm_source_files})

target_link_libraries(qm la)
