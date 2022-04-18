# see https://cmake.org/cmake/help/latest/module/FindBoost.html
set (Boost_USE_STATIC_LIBS OFF)
find_package(Boost REQUIRED COMPONENTS unit_test_framework)

include_directories(${Boost_INCLUDE_DIR})

# Note: a target should be already defined using 'add_executable' or 'add_library'
add_executable (Sparse_test_run SparseTest.cu)
target_link_libraries(Sparse_test_run la ${Boost_LIBRARIES})
