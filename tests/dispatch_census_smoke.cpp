#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "mlx/backend/common/dispatch_census.h"

namespace census = mlx::core::census;

int fail(const std::string& message) {
  std::cerr << message << '\n';
  return 1;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    return fail("usage: dispatch_census_smoke <enabled|disabled> <jsonl>");
  }
  const std::string mode = argv[1];
  const std::filesystem::path path = argv[2];
  std::filesystem::remove(path);

  if (census::parse_max_active_tasks(nullptr) != 10 ||
      census::parse_max_active_tasks("") != 10 ||
      census::parse_max_active_tasks("1") != 1 ||
      census::parse_max_active_tasks("32") != 32 ||
      census::parse_max_active_tasks("1000000") != 1000000 ||
      census::parse_max_active_tasks("0") != 10 ||
      census::parse_max_active_tasks("1000001") != 10 ||
      census::parse_max_active_tasks("1x") != 10 ||
      census::parse_max_active_tasks("-1") != 10) {
    return fail("MLX_MAX_ACTIVE_TASKS parser contract failed");
  }

  if (mode == "disabled") {
    census::note_wait("disabled_probe", 1000);
    census::flush();
    return std::filesystem::exists(path)
        ? fail("disabled census created a file")
        : 0;
  }
  if (mode != "enabled") {
    return fail("unknown mode");
  }

  const void* pipeline = reinterpret_cast<const void*>(uintptr_t{1});
  census::register_kernel(pipeline, "cpu_smoke_kernel");
  census::note_pipeline(pipeline);
  census::note_set_bytes(16);
  census::note_buffer_bind();
  census::note_dispatch(
      "threads", census::Dim3{8, 1, 1}, census::Dim3{4, 1, 1});
  const uint64_t cb = census::note_cb_encode_end();
  census::note_cb_gpu_times(cb, 100.0, 200.0);
  census::note_wait("cap_wait", 1000);
  census::flush();

  std::ifstream input(path);
  std::ostringstream buffer;
  buffer << input.rdbuf();
  const std::string jsonl = buffer.str();
  for (const std::string& required : {
           "\"record\":\"op\"",
           "\"kernel_name\":\"cpu_smoke_kernel\"",
           "\"setBytes_total_bytes\":16",
           "\"buffer_binds\":1",
           "\"record\":\"cb\"",
           "\"gpu_start_ns\":100",
           "\"gpu_end_ns\":200",
           "\"record\":\"wait\"",
           "\"bucket\":\"cap_wait\""}) {
    if (jsonl.find(required) == std::string::npos) {
      return fail("missing JSONL contract: " + required);
    }
  }
  return 0;
}
