set(models_source_files ElasticChain1D.cpp ElasticChain1D.h)
add_library(models ${models_source_files})

target_link_libraries(models qm)
