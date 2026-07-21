// Copyright © 2026 Apple Inc.
//
// S2a dispatch-stream census. See dispatch_census.h for the contract.

#include "mlx/backend/common/dispatch_census.h"

#include <atomic>
#include <cstdlib>
#include <fstream>
#include <map>
#include <mach/mach_time.h>
#include <mutex>
#include <string>
#include <unordered_map>

namespace mlx::core::census {

namespace detail {

static bool init_enabled() {
  const char* path = std::getenv("MLX_DISPATCH_CENSUS");
  return path != nullptr && path[0] != '\0';
}

// Initialised at static-init, before any Metal command is encoded.
bool g_enabled = init_enabled();

} // namespace detail

// Emit a "wait" event line only for stalls at least this long; shorter waits
// still accumulate into the per-bucket totals but do not flood the JSONL.
static constexpr uint64_t kWaitEmitFloorNs = 250;

uint64_t now_ns() {
  static mach_timebase_info_data_t tb = [] {
    mach_timebase_info_data_t t;
    mach_timebase_info(&t);
    return t;
  }();
  return mach_absolute_time() * tb.numer / tb.denom;
}

namespace {

// ---- JSONL sink ----------------------------------------------------------
std::mutex g_file_mtx;
std::ofstream g_file;
bool g_file_opened = false;
bool g_file_failed = false;

void append_json_string(std::string& out, const std::string& value) {
  out.push_back('"');
  for (char c : value) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          static const char* hex = "0123456789abcdef";
          out += "\\u00";
          out.push_back(hex[(c >> 4) & 0xF]);
          out.push_back(hex[c & 0xF]);
        } else {
          out.push_back(c);
        }
    }
  }
  out.push_back('"');
}

// Writes one complete JSONL line under the file mutex.
void write_line(const std::string& line) {
  std::lock_guard<std::mutex> lk(g_file_mtx);
  if (!g_file_opened && !g_file_failed) {
    const char* path = std::getenv("MLX_DISPATCH_CENSUS");
    if (path != nullptr && path[0] != '\0') {
      g_file.open(path, std::ios::out | std::ios::trunc);
    }
    g_file_opened = true;
    g_file_failed = !g_file.is_open();
  }
  if (!g_file_failed) {
    g_file << line;
  }
}

// ---- op / kernel-name state ----------------------------------------------
std::mutex g_name_mtx;
std::unordered_map<const void*, std::string> g_names;
std::atomic<uint64_t> g_seq{0};
std::atomic<uint64_t> g_cb_counter{0};

struct Pending {
  const void* pipeline = nullptr;
  uint32_t set_bytes_calls = 0;
  uint64_t set_bytes_total_bytes = 0;
  uint32_t buffer_binds = 0;
};
thread_local Pending g_pending;

// ---- per-command-buffer accumulator --------------------------------------
struct CbAccum {
  bool started = false;
  uint64_t uid = 0;
  uint64_t encode_start_ns = 0;
  uint64_t first_op_seq = 0;
  uint64_t last_op_seq = 0;
  uint64_t op_count = 0;
};
thread_local CbAccum g_cb;

struct CbRecord {
  uint64_t encode_start_ns = 0;
  uint64_t encode_end_ns = 0;
  uint64_t first_op_seq = 0;
  uint64_t last_op_seq = 0;
  uint64_t op_count = 0;
};
std::mutex g_cb_mtx;
std::unordered_map<uint64_t, CbRecord> g_cb_records;

// ---- wait / counter buckets ----------------------------------------------
struct Bucket {
  uint64_t count = 0;
  uint64_t total_ns = 0;
};
std::mutex g_bucket_mtx;
std::map<std::string, Bucket> g_buckets;

std::string resolve_name(const void* pipeline) {
  if (pipeline == nullptr) {
    return "<none>";
  }
  std::lock_guard<std::mutex> lk(g_name_mtx);
  auto it = g_names.find(pipeline);
  return it == g_names.end() ? std::string("<unknown>") : it->second;
}

// Dump the accumulated bucket table as one summary line.
void write_summary() {
  std::string line = "{\"record\":\"summary\",\"ops_total\":";
  line += std::to_string(g_seq.load(std::memory_order_relaxed));
  line += ",\"cbs_total\":";
  line += std::to_string(g_cb_counter.load(std::memory_order_relaxed));
  line += ",\"buckets\":{";
  {
    std::lock_guard<std::mutex> lk(g_bucket_mtx);
    bool first = true;
    for (const auto& [name, b] : g_buckets) {
      if (!first) {
        line += ",";
      }
      first = false;
      append_json_string(line, name);
      line += ":{\"count\":" + std::to_string(b.count) +
          ",\"total_ns\":" + std::to_string(b.total_ns) + "}";
    }
  }
  line += "}}\n";
  write_line(line);
}

// Flush the sink on process teardown so a run that ends without an explicit
// synchronize still leaves a complete artifact (summary + flush).
struct Closer {
  ~Closer() {
    if (detail::g_enabled) {
      write_summary();
    }
    std::lock_guard<std::mutex> lk(g_file_mtx);
    if (g_file_opened && !g_file_failed) {
      g_file.flush();
    }
  }
};
Closer g_closer;

} // namespace

void register_kernel(const void* pipeline, const char* name) {
  if (!detail::g_enabled || pipeline == nullptr || name == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lk(g_name_mtx);
  g_names.emplace(pipeline, std::string(name));
}

void note_pipeline(const void* pipeline) {
  if (!detail::g_enabled) {
    return;
  }
  g_pending.pipeline = pipeline;
}

void note_set_bytes(std::size_t nbytes) {
  if (!detail::g_enabled) {
    return;
  }
  g_pending.set_bytes_calls += 1;
  g_pending.set_bytes_total_bytes += nbytes;
}

void note_buffer_bind() {
  if (!detail::g_enabled) {
    return;
  }
  g_pending.buffer_binds += 1;
}

void note_dispatch(const char* dispatch_kind, Dim3 grid, Dim3 threadgroup) {
  if (!detail::g_enabled) {
    return;
  }

  const uint64_t seq = g_seq.fetch_add(1, std::memory_order_relaxed);

  // Start a new command-buffer accumulator on the first op after a commit.
  if (!g_cb.started) {
    g_cb.started = true;
    g_cb.uid = g_cb_counter.fetch_add(1, std::memory_order_relaxed);
    g_cb.encode_start_ns = now_ns();
    g_cb.first_op_seq = seq;
    g_cb.op_count = 0;
  }
  g_cb.last_op_seq = seq;
  g_cb.op_count += 1;

  const std::string name = resolve_name(g_pending.pipeline);

  std::string line;
  line.reserve(288);
  line += "{\"record\":\"op\",\"seq\":";
  line += std::to_string(seq);
  line += ",\"command_buffer_index\":";
  line += std::to_string(g_cb.uid);
  line += ",\"kind\":\"compute\",\"dispatch\":";
  append_json_string(line, dispatch_kind ? dispatch_kind : "");
  line += ",\"kernel_name\":";
  append_json_string(line, name);
  line += ",\"setBytes_calls\":";
  line += std::to_string(g_pending.set_bytes_calls);
  line += ",\"setBytes_total_bytes\":";
  line += std::to_string(g_pending.set_bytes_total_bytes);
  line += ",\"buffer_binds\":";
  line += std::to_string(g_pending.buffer_binds);
  line += ",\"grid\":[";
  line += std::to_string(grid.x) + "," + std::to_string(grid.y) + "," +
      std::to_string(grid.z);
  line += "],\"threadgroup\":[";
  line += std::to_string(threadgroup.x) + "," + std::to_string(threadgroup.y) +
      "," + std::to_string(threadgroup.z);
  line += "]}\n";
  write_line(line);

  // Reset per-command accumulators; the next command starts clean.
  g_pending.pipeline = nullptr;
  g_pending.set_bytes_calls = 0;
  g_pending.set_bytes_total_bytes = 0;
  g_pending.buffer_binds = 0;
}

uint64_t note_cb_encode_end() {
  if (!detail::g_enabled) {
    return 0;
  }
  const uint64_t end_ns = now_ns();
  uint64_t uid;
  CbRecord rec;
  if (g_cb.started) {
    uid = g_cb.uid;
    rec.encode_start_ns = g_cb.encode_start_ns;
    rec.first_op_seq = g_cb.first_op_seq;
    rec.last_op_seq = g_cb.last_op_seq;
    rec.op_count = g_cb.op_count;
  } else {
    // Committed with no ops encoded (e.g. an empty finalize).
    uid = g_cb_counter.fetch_add(1, std::memory_order_relaxed);
    rec.encode_start_ns = end_ns;
    rec.first_op_seq = 0;
    rec.last_op_seq = 0;
    rec.op_count = 0;
  }
  rec.encode_end_ns = end_ns;
  {
    std::lock_guard<std::mutex> lk(g_cb_mtx);
    g_cb_records[uid] = rec;
  }
  g_cb.started = false; // next op opens a fresh command buffer
  return uid;
}

void note_cb_gpu_times(
    uint64_t cb_index,
    double gpu_start_ns,
    double gpu_end_ns) {
  if (!detail::g_enabled) {
    return;
  }
  CbRecord rec;
  {
    std::lock_guard<std::mutex> lk(g_cb_mtx);
    auto it = g_cb_records.find(cb_index);
    if (it == g_cb_records.end()) {
      return;
    }
    rec = it->second;
    g_cb_records.erase(it);
  }
  std::string line = "{\"record\":\"cb\",\"command_buffer_index\":";
  line += std::to_string(cb_index);
  line += ",\"op_count\":" + std::to_string(rec.op_count);
  line += ",\"first_op_seq\":" + std::to_string(rec.first_op_seq);
  line += ",\"last_op_seq\":" + std::to_string(rec.last_op_seq);
  line += ",\"encode_start_ns\":" + std::to_string(rec.encode_start_ns);
  line += ",\"encode_end_ns\":" + std::to_string(rec.encode_end_ns);
  // GPU timestamps arrive as doubles (seconds*1e9); store as integer ns.
  line += ",\"gpu_start_ns\":" +
      std::to_string(static_cast<uint64_t>(gpu_start_ns < 0 ? 0 : gpu_start_ns));
  line += ",\"gpu_end_ns\":" +
      std::to_string(static_cast<uint64_t>(gpu_end_ns < 0 ? 0 : gpu_end_ns));
  line += "}\n";
  write_line(line);
}

void note_wait(const char* bucket, uint64_t wait_ns) {
  if (!detail::g_enabled) {
    return;
  }
  const std::string name = bucket ? bucket : "";
  {
    std::lock_guard<std::mutex> lk(g_bucket_mtx);
    auto& b = g_buckets[name];
    b.count += 1;
    b.total_ns += wait_ns;
  }
  if (wait_ns < kWaitEmitFloorNs) {
    return;
  }
  std::string line = "{\"record\":\"wait\",\"bucket\":";
  append_json_string(line, name);
  line += ",\"wait_ns\":" + std::to_string(wait_ns);
  line += ",\"at_ns\":" + std::to_string(now_ns());
  line += "}\n";
  write_line(line);
}

void flush() {
  std::lock_guard<std::mutex> lk(g_file_mtx);
  if (g_file_opened && !g_file_failed) {
    g_file.flush();
  }
}

} // namespace mlx::core::census
