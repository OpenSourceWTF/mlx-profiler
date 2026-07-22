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
set(CENSUS_MULTI ${CMAKE_CURRENT_BINARY_DIR}/dispatch-census-multi-state.jsonl)
set(CENSUS_ENCODE_START
    ${CMAKE_CURRENT_BINARY_DIR}/dispatch-census-encode-start.jsonl)
set(CENSUS_PIPELINE
    ${CMAKE_CURRENT_BINARY_DIR}/dispatch-census-pipeline-persistence.jsonl)
set(CENSUS_REREGISTER
    ${CMAKE_CURRENT_BINARY_DIR}/dispatch-census-kernel-reregister.jsonl)
set(CENSUS_INVALID
    ${CMAKE_CURRENT_BINARY_DIR}/missing-census-dir/census.jsonl)
set(CENSUS_SNAPSHOT
    ${CMAKE_CURRENT_BINARY_DIR}/dispatch-census-path-snapshot.jsonl)
set(CENSUS_FINALIZE
    ${CMAKE_CURRENT_BINARY_DIR}/dispatch-census-finalize.jsonl)
add_test(
  NAME dispatch_census_disabled
  COMMAND ${CMAKE_COMMAND} -E env --unset=MLX_DISPATCH_CENSUS
          $<TARGET_FILE:dispatch_census_smoke> disabled ${CENSUS_OFF})
add_test(
  NAME dispatch_census_enabled
  COMMAND ${CMAKE_COMMAND} -E env MLX_DISPATCH_CENSUS=${CENSUS_ON}
          $<TARGET_FILE:dispatch_census_smoke> enabled ${CENSUS_ON})
add_test(
  NAME dispatch_census_multi_state
  COMMAND ${CMAKE_COMMAND} -E env MLX_DISPATCH_CENSUS=${CENSUS_MULTI}
          $<TARGET_FILE:dispatch_census_smoke> multi_state ${CENSUS_MULTI})
add_test(
  NAME dispatch_census_encode_start
  COMMAND ${CMAKE_COMMAND} -E env MLX_DISPATCH_CENSUS=${CENSUS_ENCODE_START}
          $<TARGET_FILE:dispatch_census_smoke> encode_start
          ${CENSUS_ENCODE_START})
add_test(
  NAME dispatch_census_pipeline_persistence
  COMMAND ${CMAKE_COMMAND} -E env MLX_DISPATCH_CENSUS=${CENSUS_PIPELINE}
          $<TARGET_FILE:dispatch_census_smoke> pipeline_persistence
          ${CENSUS_PIPELINE})
add_test(
  NAME dispatch_census_kernel_reregister
  COMMAND ${CMAKE_COMMAND} -E env MLX_DISPATCH_CENSUS=${CENSUS_REREGISTER}
          $<TARGET_FILE:dispatch_census_smoke> kernel_reregister
          ${CENSUS_REREGISTER})
add_test(
  NAME dispatch_census_portable_clock
  COMMAND
    ${CMAKE_COMMAND}
    -DSOURCE=${PROJECT_SOURCE_DIR}/mlx/backend/common/dispatch_census.cpp -P
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch_census_portability.cmake)
add_test(
  NAME dispatch_census_invalid_path
  COMMAND ${CMAKE_COMMAND} -E env MLX_DISPATCH_CENSUS=${CENSUS_INVALID}
          $<TARGET_FILE:dispatch_census_smoke> invalid_path ${CENSUS_INVALID})
add_test(
  NAME dispatch_census_path_snapshot
  COMMAND ${CMAKE_COMMAND} -E env MLX_DISPATCH_CENSUS=${CENSUS_SNAPSHOT}
          $<TARGET_FILE:dispatch_census_smoke> path_snapshot ${CENSUS_SNAPSHOT})
add_test(
  NAME dispatch_census_finalize
  COMMAND ${CMAKE_COMMAND} -E env MLX_DISPATCH_CENSUS=${CENSUS_FINALIZE}
          $<TARGET_FILE:dispatch_census_smoke> finalize ${CENSUS_FINALIZE})

if(APPLE)
  set(CENSUS_FIFO ${CMAKE_CURRENT_BINARY_DIR}/dispatch-census-sink.fifo)
  add_test(
    NAME dispatch_census_nonblocking_sink
    COMMAND ${CMAKE_COMMAND} -E env MLX_DISPATCH_CENSUS=${CENSUS_FIFO}
            $<TARGET_FILE:dispatch_census_smoke> nonblocking_sink
            ${CENSUS_FIFO})
  add_test(
    NAME clt_preamble_excludes_stub_headers
    COMMAND
      bash ${CMAKE_CURRENT_SOURCE_DIR}/clt_preamble_smoke.sh
      ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/clt-preamble-smoke
      ${CMAKE_C_COMPILER})
endif()
