set(models_source_files
        ElasticChain1D.cu
        ElasticChain1D.cuh
#        XYZModel.cpp
#        XYZModel.h
        )

add_library(models ${models_source_files})

target_link_libraries(models qm)
