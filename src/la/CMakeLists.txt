set(la_source_files
        Dense.cu
        Dense.cuh
        Vect.cu
        Vect.cuh
        Sparse.cu
        Sparse.cuh
        SparseELL.cu
        SparseELL.cuh
        Super.cu
        Super.cuh
        )
add_library(la ${la_source_files})
target_link_libraries(la utils)
