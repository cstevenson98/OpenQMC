set(models_source_files
        ElasticChain1D.cu
        ElasticChain1D.cuh
        XYZModel.cu
        XYZModel.cuh
        )

add_library(models ${models_source_files})

target_link_libraries(models qm)
