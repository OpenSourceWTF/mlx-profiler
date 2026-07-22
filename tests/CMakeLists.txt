FetchContent_Declare(
  doctest
  GIT_REPOSITORY https://github.com/onqtam/doctest.git
  GIT_TAG v2.4.12)
FetchContent_MakeAvailable(doctest)

add_executable(tests ${PROJECT_SOURCE_DIR}/tests/tests.cpp)

if(MLX_BUILD_METAL OR MLX_BUILD_CUDA)
  set(METAL_TEST_SOURCES gpu_tests.cpp)
endif()

include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)

target_sources(
  tests
  PRIVATE allocator_tests.cpp
          array_tests.cpp
          arg_reduce_tests.cpp
          autograd_tests.cpp
          blas_tests.cpp
          compile_tests.cpp
          custom_vjp_tests.cpp
          creations_tests.cpp
          device_tests.cpp
          einsum_tests.cpp
          export_import_tests.cpp
          eval_tests.cpp
          fft_tests.cpp
          load_tests.cpp
          ops_tests.cpp
          random_tests.cpp
          scheduler_tests.cpp
          utils_tests.cpp
          vmap_tests.cpp
          linalg_tests.cpp
          ${METAL_TEST_SOURCES})

target_link_libraries(tests PRIVATE mlx doctest)
target_compile_options(tests PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(tests PRIVATE ${SANITIZER_LINK_FLAGS})

doctest_discover_tests(tests)
add_test(NAME tests COMMAND tests)

add_executable(dispatch_census_smoke dispatch_census_smoke.cpp)
target_link_libraries(dispatch_census_smoke PRIVATE mlx)
set(CENSUS_ON ${CMAKE_CURRENT_BINARY_DIR}/dispatch-census-enabled.jsonl)
set(CENSUS_OFF ${CMAKE_CURRENT_BINARY_DIR}/dispatch-census-disabled.jsonl)
add_test(
  NAME dispatch_census_disabled
  COMMAND ${CMAKE_COMMAND} -E env --unset=MLX_DISPATCH_CENSUS
          $<TARGET_FILE:dispatch_census_smoke> disabled ${CENSUS_OFF})
add_test(
  NAME dispatch_census_enabled
  COMMAND ${CMAKE_COMMAND} -E env MLX_DISPATCH_CENSUS=${CENSUS_ON}
          $<TARGET_FILE:dispatch_census_smoke> enabled ${CENSUS_ON})
