set(models_source_files
        ElasticChain1D.cpp
        ElasticChain1D.h
        XYZModel.cpp
        XYZModel.h
        )

add_library(models ${models_source_files})

target_link_libraries(models qm)
