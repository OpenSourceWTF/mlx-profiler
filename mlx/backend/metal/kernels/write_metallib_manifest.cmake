foreach(REQUIRED MLX_METALLIB MLX_MANIFEST MLX_SOURCE_SHA256 MLX_METAL_VERSION
                 MLX_MACOS_SDK_VERSION MLX_DEPLOYMENT_TARGET MLX_METAL_JIT
                 MLX_METAL_DEBUG MLX_BUILD_TYPE)
  if(NOT DEFINED ${REQUIRED})
    message(FATAL_ERROR "Missing manifest input: ${REQUIRED}")
  endif()
endforeach()

file(SHA256 "${MLX_METALLIB}" MLX_ARTIFACT_SHA256)
file(
  WRITE "${MLX_MANIFEST}"
  "format=1\n"
  "artifact_sha256=${MLX_ARTIFACT_SHA256}\n"
  "source_sha256=${MLX_SOURCE_SHA256}\n"
  "metal_version=${MLX_METAL_VERSION}\n"
  "macos_sdk=${MLX_MACOS_SDK_VERSION}\n"
  "deployment_target=${MLX_DEPLOYMENT_TARGET}\n"
  "metal_jit=${MLX_METAL_JIT}\n"
  "metal_debug=${MLX_METAL_DEBUG}\n"
  "build_type=${MLX_BUILD_TYPE}\n")
