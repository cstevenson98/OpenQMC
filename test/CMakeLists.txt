# testing directory 

# build test binary with gtest

find_package(GTest REQUIRED)

enable_testing()

# gather all test sources and headers 
file(GLOB_RECURSE test_sources "*.cc")
file(GLOB_RECURSE test_headers "*.h")

# message the tests to compile
message(STATUS "Tests to compile: ${test_sources}")


add_executable(
  oqmc_test
  ${test_sources}
  ${test_headers}
)

# check if the oqmc library is available and message if not
if(NOT TARGET oqmc)
  message(FATAL_ERROR "oqmc library not found")
endif()

target_link_libraries(oqmc_test oqmc GTest::Main GTest::GTest)
target_include_directories(oqmc_test PUBLIC "${PROJECT_SOURCE_DIR}/include")

include(GoogleTest)
gtest_discover_tests(oqmc_test)
