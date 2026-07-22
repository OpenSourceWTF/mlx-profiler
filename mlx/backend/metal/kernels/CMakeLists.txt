set(BASE_HEADERS
    bf16.h
    bf16_math.h
    complex.h
    defines.h
    erf.h
    expm1f.h
    fp8.h
    logging.h
    utils.h)

# Fingerprint every Metal source/header, plus the compiler-facing configuration
# that selects kernels. A prebuilt library is accepted only with a sidecar
# manifest produced from the same inputs.
file(
  GLOB_RECURSE MLX_METALLIB_SOURCE_INPUTS
  CONFIGURE_DEPENDS
  LIST_DIRECTORIES false
  "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.metal")
list(SORT MLX_METALLIB_SOURCE_INPUTS)
set(MLX_METALLIB_SOURCE_FINGERPRINT "")
foreach(SOURCE_INPUT IN LISTS MLX_METALLIB_SOURCE_INPUTS)
  file(SHA256 "${SOURCE_INPUT}" SOURCE_INPUT_SHA256)
  file(RELATIVE_PATH SOURCE_INPUT_RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
       "${SOURCE_INPUT}")
  string(APPEND MLX_METALLIB_SOURCE_FINGERPRINT
         "${SOURCE_INPUT_RELATIVE}:${SOURCE_INPUT_SHA256}\n")
endforeach()
string(SHA256 MLX_METALLIB_SOURCE_SHA256
       "${MLX_METALLIB_SOURCE_FINGERPRINT}")
set(MLX_METALLIB_MANIFEST ${MLX_METAL_PATH}/mlx.metallib.manifest)

function(build_kernel_base TARGET SRCFILE DEPS)
  # CLT-only build: no Metal compiler, the metallib is supplied prebuilt, so the
  # per-kernel .air compile is skipped entirely.
  if(MLX_METAL_PREBUILT_LIB)
    return()
  endif()
  set(METAL_FLAGS
      -x
      metal
      -Wall
      -Wextra
      -fno-fast-math
      -Wno-c++17-extensions
      -Wno-c++20-extensions)
  if(MLX_METAL_DEBUG)
    set(METAL_FLAGS ${METAL_FLAGS} -gline-tables-only -frecord-sources)
  endif()
  if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND MLX_METAL_VERSION GREATER_EQUAL 320)
    set(METAL_FLAGS ${METAL_FLAGS} -fmetal-enable-logging)
  endif()
  if(NOT CMAKE_OSX_DEPLOYMENT_TARGET STREQUAL "")
    set(METAL_FLAGS ${METAL_FLAGS}
                    "-mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")
  endif()
  add_custom_command(
    COMMAND xcrun -sdk macosx metal ${METAL_FLAGS} -c ${SRCFILE}
            -I${PROJECT_SOURCE_DIR} -o ${TARGET}.air
    DEPENDS ${SRCFILE} ${DEPS} ${BASE_HEADERS}
    OUTPUT ${TARGET}.air
    COMMENT "Building ${TARGET}.air"
    VERBATIM)
endfunction(build_kernel_base)

function(build_kernel KERNEL)
  # CLT-only build: leave KERNEL_AIR empty; the metallib is supplied prebuilt.
  if(MLX_METAL_PREBUILT_LIB)
    return()
  endif()
  set(SRCFILE ${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL}.metal)
  cmake_path(GET KERNEL STEM TARGET)
  build_kernel_base(${TARGET} ${SRCFILE} "${ARGN}")
  set(KERNEL_AIR
      ${TARGET}.air ${KERNEL_AIR}
      PARENT_SCOPE)
endfunction(build_kernel)

build_kernel(arg_reduce)
build_kernel(conv steel/conv/params.h)
build_kernel(layer_norm)
build_kernel(random)
build_kernel(rms_norm)
build_kernel(rope)
build_kernel(scaled_dot_product_attention sdpa_vector.h)
if(MLX_METAL_VERSION GREATER_EQUAL 320)
  build_kernel(fence)
endif()

set(STEEL_HEADERS
    steel/defines.h
    steel/utils.h
    steel/conv/conv.h
    steel/conv/loader.h
    steel/conv/loaders/loader_channel_l.h
    steel/conv/loaders/loader_channel_n.h
    steel/conv/loaders/loader_general.h
    steel/conv/kernels/steel_conv.h
    steel/conv/kernels/steel_conv_3d.h
    steel/conv/kernels/steel_conv_general.h
    steel/gemm/gemm.h
    steel/gemm/mma.h
    steel/gemm/loader.h
    steel/gemm/params.h
    steel/gemm/transforms.h
    steel/gemm/kernels/steel_gemm_fused.h
    steel/gemm/kernels/steel_gemm_gather.h
    steel/gemm/kernels/steel_gemm_masked.h
    steel/gemm/kernels/steel_gemm_segmented.h
    steel/gemm/kernels/steel_gemm_splitk.h
    steel/utils/type_traits.h
    steel/utils/integral_constant.h)

set(STEEL_ATTN_HEADERS
    steel/defines.h
    steel/utils.h
    steel/gemm/gemm.h
    steel/gemm/mma.h
    steel/gemm/loader.h
    steel/gemm/transforms.h
    steel/utils/type_traits.h
    steel/utils/integral_constant.h
    steel/attn/attn.h
    steel/attn/loader.h
    steel/attn/mma.h
    steel/attn/params.h
    steel/attn/transforms.h
    steel/attn/kernels/steel_attention.h)

set(STEEL_NAX_HEADERS
    steel/defines.h
    steel/utils.h
    steel/gemm/params.h
    steel/gemm/transforms.h
    steel/gemm/nax.h
    steel/gemm/gemm_nax.h
    steel/utils/type_traits.h
    steel/utils/integral_constant.h
    steel/gemm/kernels/steel_gemm_fused_nax.h
    steel/gemm/kernels/steel_gemm_gather_nax.h
    steel/gemm/kernels/steel_gemm_splitk_nax.h
    steel/gemm/kernels/steel_gemm_segmented_nax.h)

set(STEEL_NAX_ATTN_HEADERS
    steel/defines.h
    steel/utils.h
    steel/attn/nax.h
    steel/utils/type_traits.h
    steel/utils/integral_constant.h
    steel/attn/params.h
    steel/attn/kernels/steel_attention_nax.h)

if(NOT MLX_METAL_JIT)
  build_kernel(arange arange.h)
  build_kernel(binary binary.h binary_ops.h)
  build_kernel(binary_two binary_two.h)
  build_kernel(copy copy.h)
  build_kernel(fft fft.h fft/radix.h fft/readwrite.h)
  build_kernel(
    reduce
    atomic.h
    reduction/ops.h
    reduction/reduce_init.h
    reduction/reduce_all.h
    reduction/reduce_col.h
    reduction/reduce_row.h)
  build_kernel(quantized quantized.h quantized_utils.h ${STEEL_HEADERS})
  build_kernel(fp_quantized fp4.h fp8.h fp_quantized.h quantized_utils.h
               ${STEEL_HEADERS})
  build_kernel(scan scan.h)
  build_kernel(softmax softmax.h)
  build_kernel(logsumexp logsumexp.h)
  build_kernel(sort sort.h)
  build_kernel(ternary ternary.h ternary_ops.h)
  build_kernel(unary unary.h unary_ops.h)
  build_kernel(steel/conv/kernels/steel_conv ${STEEL_HEADERS})
  build_kernel(steel/conv/kernels/steel_conv_3d ${STEEL_HEADERS})
  build_kernel(steel/conv/kernels/steel_conv_general ${STEEL_HEADERS})
  build_kernel(steel/gemm/kernels/steel_gemm_fused ${STEEL_HEADERS})
  build_kernel(steel/gemm/kernels/steel_gemm_gather ${STEEL_HEADERS})
  build_kernel(steel/gemm/kernels/steel_gemm_masked ${STEEL_HEADERS})
  build_kernel(steel/gemm/kernels/steel_gemm_splitk ${STEEL_HEADERS})
  build_kernel(steel/gemm/kernels/steel_gemm_segmented ${STEEL_HEADERS})
  build_kernel(gemv steel/utils.h)
  build_kernel(gemv_masked steel/utils.h)
  build_kernel(steel/attn/kernels/steel_attention ${STEEL_ATTN_HEADERS})

  if(MLX_METAL_VERSION GREATER_EQUAL 400
     AND MACOS_SDK_VERSION VERSION_GREATER_EQUAL 26.2
     AND CMAKE_OSX_DEPLOYMENT_TARGET VERSION_GREATER_EQUAL 26.2)

    build_kernel(steel/gemm/kernels/steel_gemm_fused_nax ${STEEL_NAX_HEADERS})
    build_kernel(steel/gemm/kernels/steel_gemm_gather_nax ${STEEL_NAX_HEADERS})
    build_kernel(steel/gemm/kernels/steel_gemm_splitk_nax ${STEEL_NAX_HEADERS})
    build_kernel(steel/gemm/kernels/steel_gemm_segmented_nax
                 ${STEEL_NAX_HEADERS})

    build_kernel(quantized_nax quantized_nax.h ${STEEL_NAX_HEADERS})
    build_kernel(fp_quantized_nax fp4.h fp8.h fp_quantized_nax.h
                 ${STEEL_NAX_HEADERS})

    build_kernel(steel/attn/kernels/steel_attention_nax
                 ${STEEL_NAX_ATTN_HEADERS})

  else()
    message(
      WARNING "NAX kernels require Metal 4, macOS SDK >= 26.2, and "
              "MACOSX_DEPLOYMENT_TARGET >= 26.2 (SDK ${MACOS_SDK_VERSION}, "
              "CMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}). "
              "Building without NAX kernels.")
    target_compile_definitions(mlx PRIVATE MLX_METAL_NO_NAX)
  endif()

endif()

set(METAL_LINK_FLAGS)
if(NOT CMAKE_OSX_DEPLOYMENT_TARGET STREQUAL "")
  set(METAL_LINK_FLAGS "-mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")
endif()

if(MLX_METAL_PREBUILT_LIB)
  # Reuse a prebuilt mlx.metallib (no Metal compiler on this box). Copy it into
  # the location baked into METAL_PATH so both the compile-time fallback path in
  # device.cpp and the install rule below resolve it. current_binary_dir()
  # (dladdr on libmlx) colocates it next to libmlx.dylib after install.
  if(NOT EXISTS "${MLX_METAL_PREBUILT_LIB}")
    message(
      FATAL_ERROR
        "MLX_METAL_PREBUILT_LIB does not exist: ${MLX_METAL_PREBUILT_LIB}")
  endif()
  set(MLX_METAL_PREBUILT_MANIFEST
      "${MLX_METAL_PREBUILT_LIB}.manifest")
  if(NOT EXISTS "${MLX_METAL_PREBUILT_MANIFEST}")
    message(
      FATAL_ERROR
        "MLX_METAL_PREBUILT_LIB requires an adjacent source manifest: "
        "${MLX_METAL_PREBUILT_MANIFEST}")
  endif()
  file(READ "${MLX_METAL_PREBUILT_MANIFEST}" PREBUILT_MANIFEST)
  file(SHA256 "${MLX_METAL_PREBUILT_LIB}" PREBUILT_ARTIFACT_SHA256)
  foreach(MANIFEST_FIELD
          artifact_sha256
          source_sha256
          metal_version
          macos_sdk
          deployment_target
          metal_jit
          metal_debug
          build_type)
    unset(CMAKE_MATCH_1)
    string(REGEX MATCH "${MANIFEST_FIELD}=([^\n]+)" _MATCH
                 "${PREBUILT_MANIFEST}")
    if("${CMAKE_MATCH_1}" STREQUAL "")
      message(
        FATAL_ERROR
          "Invalid metallib manifest: missing ${MANIFEST_FIELD} in "
          "${MLX_METAL_PREBUILT_MANIFEST}")
    endif()
    set(PREBUILT_${MANIFEST_FIELD} "${CMAKE_MATCH_1}")
  endforeach()
  if(NOT PREBUILT_artifact_sha256 STREQUAL PREBUILT_ARTIFACT_SHA256
     OR NOT PREBUILT_source_sha256 STREQUAL MLX_METALLIB_SOURCE_SHA256
     OR NOT PREBUILT_metal_version STREQUAL "${MLX_METAL_VERSION}"
     OR NOT PREBUILT_macos_sdk STREQUAL "${MACOS_SDK_VERSION}"
     OR NOT PREBUILT_deployment_target STREQUAL
            "${CMAKE_OSX_DEPLOYMENT_TARGET}"
     OR NOT PREBUILT_metal_jit STREQUAL "${MLX_METAL_JIT}"
     OR NOT PREBUILT_metal_debug STREQUAL "${MLX_METAL_DEBUG}"
     OR NOT PREBUILT_build_type STREQUAL "${CMAKE_BUILD_TYPE}")
    message(
      FATAL_ERROR
        "MLX_METAL_PREBUILT_LIB manifest does not match this source/build: "
        "${MLX_METAL_PREBUILT_MANIFEST}")
  endif()
  configure_file(${MLX_METAL_PREBUILT_LIB} ${MLX_METAL_PATH}/mlx.metallib
                 COPYONLY)
  configure_file(${MLX_METAL_PREBUILT_MANIFEST} ${MLX_METALLIB_MANIFEST}
                 COPYONLY)
else()
  add_custom_command(
    OUTPUT ${MLX_METAL_PATH}/mlx.metallib ${MLX_METALLIB_MANIFEST}
    COMMAND xcrun -sdk macosx metal ${METAL_LINK_FLAGS} ${KERNEL_AIR} -o
          ${MLX_METAL_PATH}/mlx.metallib
    COMMAND
      ${CMAKE_COMMAND}
      "-DMLX_METALLIB=${MLX_METAL_PATH}/mlx.metallib"
      "-DMLX_MANIFEST=${MLX_METALLIB_MANIFEST}"
      "-DMLX_SOURCE_SHA256=${MLX_METALLIB_SOURCE_SHA256}"
      "-DMLX_METAL_VERSION=${MLX_METAL_VERSION}"
      "-DMLX_MACOS_SDK_VERSION=${MACOS_SDK_VERSION}"
      "-DMLX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}"
      "-DMLX_METAL_JIT=${MLX_METAL_JIT}"
      "-DMLX_METAL_DEBUG=${MLX_METAL_DEBUG}"
      "-DMLX_BUILD_TYPE=${CMAKE_BUILD_TYPE}" -P
      ${CMAKE_CURRENT_SOURCE_DIR}/write_metallib_manifest.cmake
    DEPENDS ${KERNEL_AIR}
    COMMENT "Building mlx.metallib"
    VERBATIM)

  add_custom_target(
    mlx-metallib
    DEPENDS ${MLX_METAL_PATH}/mlx.metallib ${MLX_METALLIB_MANIFEST})

  add_dependencies(mlx mlx-metallib)
endif()

# Install metallib
include(GNUInstallDirs)

install(
  FILES ${MLX_METAL_PATH}/mlx.metallib ${MLX_METALLIB_MANIFEST}
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  COMPONENT metallib)
