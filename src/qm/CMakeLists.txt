set(qm_source_files
        Spins.cu
        Spins.cuh
        ../models/ElasticChain1D.cu
        ../models/ElasticChain1D.cuh
        ../models/XYZModel.cu
        ../models/XYZModel.cuh
        )

add_library(qm ${qm_source_files})

target_link_libraries(qm la)
