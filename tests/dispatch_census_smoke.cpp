#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

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
  if (mode == "invalid_path") {
    try {
      census::initialize();
    } catch (const std::runtime_error& error) {
      return std::string(error.what()).find("MLX_DISPATCH_CENSUS") ==
              std::string::npos
          ? fail("invalid census path failed without a clear diagnostic")
          : 0;
    }
    return fail("invalid census path was silently accepted");
  }
  if (mode == "path_snapshot") {
    const std::filesystem::path alternate = path.string() + ".alternate";
    std::filesystem::remove(alternate);
    setenv("MLX_DISPATCH_CENSUS", alternate.c_str(), 1);
    census::initialize();
    census::note_wait("snapshot_probe", 1000);
    census::flush();
    if (!std::filesystem::exists(path) ||
        std::filesystem::exists(alternate)) {
      return fail("census enable flag and output path were not one snapshot");
    }
    return 0;
  }
#if !defined(_WIN32)
  if (mode == "nonblocking_sink") {
    if (mkfifo(path.c_str(), 0600) != 0) {
      return fail("failed to create census sink FIFO");
    }
    const int reader = open(path.c_str(), O_RDONLY | O_NONBLOCK);
    if (reader < 0) {
      return fail("failed to open census sink FIFO reader");
    }
    census::initialize();

    std::atomic<bool> producer_done{false};
    std::thread producer([&] {
      constexpr const char* bucket =
          "blocked_sink_payload_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";
      for (int i = 0; i < 5000; ++i) {
        census::note_wait(bucket, 1000);
      }
      producer_done.store(true, std::memory_order_release);
    });

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
    while (!producer_done.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const bool producer_blocked =
        !producer_done.load(std::memory_order_acquire);

    std::atomic<bool> stop_reader{false};
    std::thread drain([&] {
      char buffer[4096];
      while (!stop_reader.load(std::memory_order_acquire)) {
        const ssize_t count = read(reader, buffer, sizeof(buffer));
        if (count < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
          break;
        }
        if (count <= 0) {
          std::this_thread::yield();
        }
      }
    });

    producer.join();
    census::finalize();
    stop_reader.store(true, std::memory_order_release);
    drain.join();
    close(reader);
    std::filesystem::remove(path);
    return producer_blocked
        ? fail("census producer blocked on sink I/O")
        : 0;
  }
#endif

  census::initialize();
  if (mode == "multi_state") {
    census::CommandBufferState first;
    census::CommandBufferState second;
    const void* first_pipeline = reinterpret_cast<const void*>(uintptr_t{1});
    const void* second_pipeline = reinterpret_cast<const void*>(uintptr_t{2});
    census::register_kernel(first_pipeline, "first_kernel");
    census::register_kernel(second_pipeline, "second_kernel");

    census::note_pipeline(first, first_pipeline);
    census::note_dispatch(
        first, "threads", census::Dim3{8, 1, 1}, census::Dim3{4, 1, 1});
    census::note_pipeline(second, second_pipeline);
    census::note_dispatch(
        second, "threads", census::Dim3{16, 1, 1}, census::Dim3{8, 1, 1});

    const uint64_t first_cb = census::note_cb_encode_end(first);
    const uint64_t second_cb = census::note_cb_encode_end(second);
    census::note_cb_gpu_times(first_cb, 100.0, 200.0);
    census::note_cb_gpu_times(second_cb, 300.0, 400.0);
    census::flush();

    std::ifstream input(path);
    std::ostringstream buffer;
    buffer << input.rdbuf();
    const std::string jsonl = buffer.str();
    for (const std::string& required : {
             "\"seq\":0,\"command_buffer_index\":0",
             "\"seq\":1,\"command_buffer_index\":1",
             "\"command_buffer_index\":0,\"op_count\":1",
             "\"command_buffer_index\":1,\"op_count\":1"}) {
      if (jsonl.find(required) == std::string::npos) {
        return fail("multi-state census collision: " + required);
      }
    }
    return 0;
  }
  if (mode == "encode_start") {
    census::CommandBufferState state;
    const void* pipeline = reinterpret_cast<const void*>(uintptr_t{3});
    const uint64_t before = census::now_ns();
    census::note_pipeline(state, pipeline);
    const uint64_t after = census::now_ns();
    if (!state.started || state.encode_start_ns < before ||
        state.encode_start_ns > after) {
      return fail("encode interval did not start at the first encoder command");
    }
    return 0;
  }
  if (mode == "pipeline_persistence") {
    census::CommandBufferState state;
    const void* pipeline = reinterpret_cast<const void*>(uintptr_t{4});
    census::register_kernel(pipeline, "persistent_kernel");
    census::note_pipeline(state, pipeline);
    census::note_dispatch(
        state, "threads", census::Dim3{8, 1, 1}, census::Dim3{4, 1, 1});
    census::note_dispatch(
        state, "threads", census::Dim3{8, 1, 1}, census::Dim3{4, 1, 1});
    const uint64_t cb = census::note_cb_encode_end(state);
    census::note_cb_gpu_times(cb, 100.0, 200.0);
    census::flush();

    std::ifstream input(path);
    std::ostringstream buffer;
    buffer << input.rdbuf();
    const std::string jsonl = buffer.str();
    const std::string kernel = "\"kernel_name\":\"persistent_kernel\"";
    const size_t first = jsonl.find(kernel);
    if (first == std::string::npos ||
        jsonl.find(kernel, first + kernel.size()) == std::string::npos) {
      return fail("pipeline state did not persist across dispatches");
    }
    return 0;
  }
  if (mode == "kernel_reregister") {
    census::CommandBufferState state;
    const void* pipeline = reinterpret_cast<const void*>(uintptr_t{5});
    census::register_kernel(pipeline, "stale_kernel");
    census::register_kernel(pipeline, "replacement_kernel");
    census::note_pipeline(state, pipeline);
    census::note_dispatch(
        state, "threads", census::Dim3{8, 1, 1}, census::Dim3{4, 1, 1});
    const uint64_t cb = census::note_cb_encode_end(state);
    census::note_cb_gpu_times(cb, 100.0, 200.0);
    census::flush();

    std::ifstream input(path);
    std::ostringstream buffer;
    buffer << input.rdbuf();
    const std::string jsonl = buffer.str();
    return jsonl.find("\"kernel_name\":\"replacement_kernel\"") ==
            std::string::npos
        ? fail("re-registered pipeline retained its stale name")
        : 0;
  }
  if (mode == "finalize") {
    std::atomic<bool> started{false};
    std::atomic<bool> run{true};
    std::thread producer([&] {
      started.store(true, std::memory_order_release);
      while (run.load(std::memory_order_acquire)) {
        census::note_wait("concurrent_probe", 0);
      }
    });
    while (!started.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    census::finalize();
    run.store(false, std::memory_order_release);
    producer.join();
    census::note_wait("after_finalize", 1000);
    census::flush();

    std::ifstream input(path);
    std::ostringstream buffer;
    buffer << input.rdbuf();
    const std::string jsonl = buffer.str();
    const std::string final_marker = "\"final\":true";
    const size_t first_final = jsonl.find(final_marker);
    if (first_final == std::string::npos ||
        jsonl.find(final_marker, first_final + final_marker.size()) !=
            std::string::npos ||
        jsonl.find("after_finalize") != std::string::npos ||
        jsonl.find('\n', first_final) + 1 != jsonl.size()) {
      return fail("census finalization was not terminal and idempotent");
    }
    return 0;
  }
  if (mode != "enabled") {
    return fail("unknown mode");
  }

  census::CommandBufferState state;
  const void* pipeline = reinterpret_cast<const void*>(uintptr_t{1});
  census::register_kernel(pipeline, "cpu_smoke_kernel");
  census::note_pipeline(state, pipeline);
  census::note_set_bytes(state, 16);
  census::note_buffer_bind(state);
  census::note_dispatch(
      state, "threads", census::Dim3{8, 1, 1}, census::Dim3{4, 1, 1});
  const uint64_t cb = census::note_cb_encode_end(state);
  census::note_cb_gpu_times(cb, 100.0, 200.0);
  census::note_wait("cap_wait", 1000);
  census::flush();

  std::ifstream input(path);
  std::ostringstream buffer;
  buffer << input.rdbuf();
  const std::string jsonl = buffer.str();
  const std::string schema = "\"schema_version\":1";
  size_t schema_count = 0;
  for (size_t pos = jsonl.find(schema); pos != std::string::npos;
       pos = jsonl.find(schema, pos + schema.size())) {
    ++schema_count;
  }
  if (schema_count != 4) {
    return fail("every emitted JSONL row must carry schema_version=1");
  }
  for (const std::string& required : {
           "\"record\":\"op\"",
           "\"kernel_name\":\"cpu_smoke_kernel\"",
           "\"setBytes_total_bytes\":16",
           "\"buffer_binds\":1",
           "\"record\":\"cb\"",
           "\"gpu_start_ns\":100",
           "\"gpu_end_ns\":200",
           "\"record\":\"wait\"",
           "\"bucket\":\"cap_wait\"",
           "\"record\":\"summary\"",
           "\"final\":false"}) {
    if (jsonl.find(required) == std::string::npos) {
      return fail("missing JSONL contract: " + required);
    }
  }
  return 0;
}
