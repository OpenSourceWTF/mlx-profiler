// Copyright © 2026 Apple Inc.
//
// S2a dispatch-stream census. See dispatch_census.h for the contract.

#include "mlx/backend/common/dispatch_census.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
#if defined(__APPLE__)
#include <mach/mach_time.h>
#endif
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>

namespace mlx::core::census {

namespace detail {

static std::string init_path() {
  const char* path = std::getenv("MLX_DISPATCH_CENSUS");
  return path == nullptr ? std::string() : std::string(path);
}

const std::string& g_path = *new std::string(init_path());
// Initialised at static-init, before any Metal command is encoded.
bool g_enabled = !g_path.empty();
#if defined(MLX_DISPATCH_CENSUS_TESTING)
std::atomic<bool> g_writer_paused{false};
#endif

} // namespace detail

// Emit a "wait" event line only for stalls at least this long; shorter waits
// still accumulate into the per-bucket totals but do not flood the JSONL.
static constexpr uint64_t kWaitEmitFloorNs = 250;
static constexpr const char* kSchemaPrefix = "{\"schema_version\":1,";
#if defined(MLX_DISPATCH_CENSUS_TESTING)
static constexpr std::size_t kMaxQueuedLines = 4;
#else
// Preserve a useful burst without allowing a stalled filesystem to consume
// unbounded process memory. Overflow is explicit in every later summary.
static constexpr std::size_t kMaxQueuedLines = 65536;
#endif

int parse_max_active_tasks(const char* value) {
  if (value != nullptr && value[0] != '\0') {
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end != value && *end == '\0' && parsed >= 1 && parsed <= 1000000) {
      return static_cast<int>(parsed);
    }
  }
  return 10;
}

uint64_t now_ns() {
#if defined(__APPLE__)
  static mach_timebase_info_data_t tb = [] {
    mach_timebase_info_data_t t;
    mach_timebase_info(&t);
    return t;
  }();
  return mach_absolute_time() * tb.numer / tb.denom;
#else
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
#endif
}

namespace {

struct CbRecord {
  uint64_t encode_start_ns = 0;
  uint64_t encode_end_ns = 0;
  uint64_t first_op_seq = 0;
  uint64_t last_op_seq = 0;
  uint64_t op_count = 0;
};

struct Bucket {
  uint64_t count = 0;
  uint64_t total_ns = 0;
};

struct QueuedLine {
  uint64_t id;
  std::string text;
};

// MLX intentionally leaks its scheduler and Metal device because their worker
// threads and callbacks may still run during static destruction. Census state
// follows the same process-lifetime rule so those producers never touch
// destructed mutexes, maps, or streams.
struct CensusState {
  std::mutex file_mtx;
  std::ofstream file;
  bool file_opened = false;
  std::once_flag file_init;
  std::mutex queue_mtx;
  std::condition_variable queue_cv;
  std::condition_variable drained_cv;
  std::condition_variable queue_space_cv;
  std::deque<QueuedLine> lines;
  uint64_t next_line_id = 0;
  uint64_t written_line_id = 0;
  std::atomic<uint64_t> dropped_rows{0};
  bool writer_stop = false;
  std::thread writer;

  std::mutex name_mtx;
  std::unordered_map<const void*, std::string> names;
  std::atomic<uint64_t> seq{0};
  std::atomic<uint64_t> cb_counter{0};
  std::atomic<uint64_t> summary_counter{0};

  std::mutex cb_mtx;
  std::unordered_map<uint64_t, CbRecord> cb_records;

  std::mutex bucket_mtx;
  std::map<std::string, Bucket> buckets;

  std::atomic<bool> closing{false};
  std::atomic<uint64_t> active_producers{0};
  std::mutex producer_mtx;
  std::condition_variable producer_cv;
};

CensusState& global_state() {
  static CensusState* state = new CensusState;
  return *state;
}

void leave_producer(CensusState& state) {
  if (state.active_producers.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    std::lock_guard<std::mutex> lk(state.producer_mtx);
    state.producer_cv.notify_all();
  }
}

class ProducerGuard {
 public:
  ProducerGuard() {
    if (!detail::g_enabled) {
      return;
    }
    auto& state = global_state();
    if (state.closing.load(std::memory_order_acquire)) {
      return;
    }
    state.active_producers.fetch_add(1, std::memory_order_acq_rel);
    if (state.closing.load(std::memory_order_acquire)) {
      leave_producer(state);
      return;
    }
    state_ = &state;
  }

  ~ProducerGuard() {
    if (state_ != nullptr) {
      leave_producer(*state_);
    }
  }

  explicit operator bool() const {
    return state_ != nullptr;
  }

  CensusState& state() const {
    return *state_;
  }

 private:
  CensusState* state_ = nullptr;
};

void writer_loop(CensusState& state) {
  while (true) {
    QueuedLine line;
    {
      std::unique_lock<std::mutex> lk(state.queue_mtx);
      state.queue_cv.wait(
          lk, [&state] {
#if defined(MLX_DISPATCH_CENSUS_TESTING)
            const bool paused =
                detail::g_writer_paused.load(std::memory_order_acquire);
#else
            constexpr bool paused = false;
#endif
            return state.writer_stop || (!paused && !state.lines.empty());
          });
      if (state.lines.empty() && state.writer_stop) {
        return;
      }
      line = std::move(state.lines.front());
      state.lines.pop_front();
    }
    state.queue_space_cv.notify_all();
    {
      std::lock_guard<std::mutex> lk(state.file_mtx);
      state.file << line.text;
    }
    {
      std::lock_guard<std::mutex> lk(state.queue_mtx);
      state.written_line_id = line.id;
    }
    state.drained_cv.notify_all();
  }
}

void initialize_file(CensusState& state) {
  std::error_code ec;
  if (std::filesystem::exists(detail::g_path, ec) &&
      !std::filesystem::is_regular_file(detail::g_path, ec)) {
    throw std::runtime_error(
        "MLX_DISPATCH_CENSUS output must be a regular file: " +
        detail::g_path);
  }
  state.file.open(detail::g_path, std::ios::out | std::ios::trunc);
  state.file_opened = state.file.is_open();
  if (!state.file_opened) {
    throw std::runtime_error(
        "MLX_DISPATCH_CENSUS cannot open output path: " + detail::g_path);
  }
  if (!std::filesystem::is_regular_file(detail::g_path, ec)) {
    state.file.close();
    state.file_opened = false;
    throw std::runtime_error(
        "MLX_DISPATCH_CENSUS output must be a regular file: " +
        detail::g_path);
  }
  state.writer = std::thread([&state] { writer_loop(state); });
}

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

// Queue one complete JSONL line. Only the dedicated writer touches the sink.
uint64_t write_line(CensusState& state, std::string line, bool barrier = false) {
  uint64_t id;
  {
    std::unique_lock<std::mutex> lk(state.queue_mtx);
    if (barrier) {
      state.queue_space_cv.wait(
          lk, [&state] { return state.lines.size() < kMaxQueuedLines; });
    } else if (state.lines.size() >= kMaxQueuedLines) {
      state.dropped_rows.fetch_add(1, std::memory_order_relaxed);
      return 0;
    }
    id = ++state.next_line_id;
    state.lines.push_back(QueuedLine{id, std::move(line)});
  }
  state.queue_cv.notify_one();
  return id;
}

void wait_until_written(CensusState& state, uint64_t line_id) {
  std::unique_lock<std::mutex> lk(state.queue_mtx);
  state.drained_cv.wait(
      lk, [&state, line_id] { return state.written_line_id >= line_id; });
}

std::string resolve_name(CensusState& state, const void* pipeline) {
  if (pipeline == nullptr) {
    return "<none>";
  }
  std::lock_guard<std::mutex> lk(state.name_mtx);
  auto it = state.names.find(pipeline);
  return it == state.names.end() ? std::string("<unknown>") : it->second;
}

void note_encode_begin(CensusState& global, CommandBufferState& encoder) {
  if (!encoder.started) {
    encoder.started = true;
    encoder.uid = global.cb_counter.fetch_add(1, std::memory_order_relaxed);
    encoder.encode_start_ns = now_ns();
    encoder.first_op_seq = 0;
    encoder.last_op_seq = 0;
    encoder.op_count = 0;
  }
}

// Dump the accumulated bucket table as one summary line.
uint64_t write_summary(CensusState& state, bool final) {
  std::string line = kSchemaPrefix;
  line += "\"record\":\"summary\",\"summary_seq\":";
  line += std::to_string(
      state.summary_counter.fetch_add(1, std::memory_order_relaxed));
  line += ",\"final\":";
  line += final ? "true" : "false";
  line += ",\"ops_total\":";
  line += std::to_string(state.seq.load(std::memory_order_relaxed));
  line += ",\"cbs_total\":";
  line += std::to_string(state.cb_counter.load(std::memory_order_relaxed));
  const uint64_t dropped =
      state.dropped_rows.load(std::memory_order_relaxed);
  line += ",\"dropped_rows\":" + std::to_string(dropped);
  line += ",\"complete\":";
  line += dropped == 0 ? "true" : "false";
  line += ",\"buckets\":{";
  {
    std::lock_guard<std::mutex> lk(state.bucket_mtx);
    bool first = true;
    for (const auto& [name, b] : state.buckets) {
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
  return write_line(state, std::move(line), true);
}

// Close the sink on process teardown without destroying callback-visible state.
// This is a terminal snapshot; it deliberately does not force a GPU drain.
struct Closer {
  ~Closer() {
    finalize();
  }
};
Closer g_closer;

} // namespace

#if defined(MLX_DISPATCH_CENSUS_TESTING)
void detail::set_writer_paused_for_test(bool paused) {
  g_writer_paused.store(paused, std::memory_order_release);
  global_state().queue_cv.notify_all();
}
#endif

void initialize() {
  if (!detail::g_enabled) {
    return;
  }
  auto& state = global_state();
  if (state.closing.load(std::memory_order_acquire)) {
    return;
  }
  std::call_once(state.file_init, [&state] { initialize_file(state); });
}

void finalize() {
  if (!detail::g_enabled) {
    return;
  }
  auto& state = global_state();
  bool expected = false;
  if (!state.closing.compare_exchange_strong(
          expected, true, std::memory_order_acq_rel)) {
    return;
  }
  {
    std::unique_lock<std::mutex> lk(state.producer_mtx);
    state.producer_cv.wait(lk, [&state] {
      return state.active_producers.load(std::memory_order_acquire) == 0;
    });
  }
  if (!state.file_opened) {
    return;
  }
  const uint64_t final_line = write_summary(state, true);
  wait_until_written(state, final_line);
  {
    std::lock_guard<std::mutex> lk(state.queue_mtx);
    state.writer_stop = true;
  }
  state.queue_cv.notify_one();
  state.writer.join();
  std::lock_guard<std::mutex> lk(state.file_mtx);
  state.file.flush();
}

void register_kernel(const void* pipeline, const char* name) {
  ProducerGuard guard;
  if (!guard || pipeline == nullptr || name == nullptr) {
    return;
  }
  auto& global = guard.state();
  std::lock_guard<std::mutex> lk(global.name_mtx);
  global.names.insert_or_assign(pipeline, std::string(name));
}

void unregister_kernel(const void* pipeline) {
  ProducerGuard guard;
  if (!guard || pipeline == nullptr) {
    return;
  }
  auto& global = guard.state();
  std::lock_guard<std::mutex> lk(global.name_mtx);
  global.names.erase(pipeline);
}

void note_pipeline(CommandBufferState& state, const void* pipeline) {
  ProducerGuard guard;
  if (!guard) {
    return;
  }
  note_encode_begin(guard.state(), state);
  state.pipeline = pipeline;
}

void note_set_bytes(CommandBufferState& state, std::size_t nbytes) {
  ProducerGuard guard;
  if (!guard) {
    return;
  }
  note_encode_begin(guard.state(), state);
  state.set_bytes_calls += 1;
  state.set_bytes_total_bytes += nbytes;
}

void note_buffer_bind(CommandBufferState& state) {
  ProducerGuard guard;
  if (!guard) {
    return;
  }
  note_encode_begin(guard.state(), state);
  state.buffer_binds += 1;
}

void note_dispatch(
    CommandBufferState& state,
    const char* dispatch_kind,
    Dim3 grid,
    Dim3 threadgroup) {
  ProducerGuard guard;
  if (!guard) {
    return;
  }
  auto& global = guard.state();

  const uint64_t seq = global.seq.fetch_add(1, std::memory_order_relaxed);

  note_encode_begin(global, state);
  if (state.op_count == 0) {
    state.first_op_seq = seq;
  }
  state.last_op_seq = seq;
  state.op_count += 1;

  const std::string name = resolve_name(global, state.pipeline);

  std::string line = kSchemaPrefix;
  line.reserve(288);
  line += "\"record\":\"op\",\"seq\":";
  line += std::to_string(seq);
  line += ",\"command_buffer_index\":";
  line += std::to_string(state.uid);
  line += ",\"kind\":\"compute\",\"dispatch\":";
  append_json_string(line, dispatch_kind ? dispatch_kind : "");
  line += ",\"kernel_name\":";
  append_json_string(line, name);
  line += ",\"setBytes_calls\":";
  line += std::to_string(state.set_bytes_calls);
  line += ",\"setBytes_total_bytes\":";
  line += std::to_string(state.set_bytes_total_bytes);
  line += ",\"buffer_binds\":";
  line += std::to_string(state.buffer_binds);
  line += ",\"grid\":[";
  line += std::to_string(grid.x) + "," + std::to_string(grid.y) + "," +
      std::to_string(grid.z);
  line += "],\"threadgroup\":[";
  line += std::to_string(threadgroup.x) + "," + std::to_string(threadgroup.y) +
      "," + std::to_string(threadgroup.z);
  line += "]}\n";
  write_line(global, std::move(line));

  // Reset per-command accumulators; the next command starts clean.
  state.set_bytes_calls = 0;
  state.set_bytes_total_bytes = 0;
  state.buffer_binds = 0;
}

uint64_t note_cb_encode_end(CommandBufferState& state) {
  ProducerGuard guard;
  if (!guard) {
    return 0;
  }
  auto& global = guard.state();
  const uint64_t end_ns = now_ns();
  uint64_t uid;
  CbRecord rec;
  if (state.started) {
    uid = state.uid;
    rec.encode_start_ns = state.encode_start_ns;
    rec.first_op_seq = state.first_op_seq;
    rec.last_op_seq = state.last_op_seq;
    rec.op_count = state.op_count;
  } else {
    // Committed with no ops encoded (e.g. an empty finalize).
    uid = global.cb_counter.fetch_add(1, std::memory_order_relaxed);
    rec.encode_start_ns = end_ns;
    rec.first_op_seq = 0;
    rec.last_op_seq = 0;
    rec.op_count = 0;
  }
  rec.encode_end_ns = end_ns;
  {
    std::lock_guard<std::mutex> lk(global.cb_mtx);
    global.cb_records[uid] = rec;
  }
  state.started = false; // next op opens a fresh command buffer
  return uid;
}

void note_cb_gpu_times(
    uint64_t cb_index,
    double gpu_start_ns,
    double gpu_end_ns) {
  ProducerGuard guard;
  if (!guard) {
    return;
  }
  auto& global = guard.state();
  CbRecord rec;
  {
    std::lock_guard<std::mutex> lk(global.cb_mtx);
    auto it = global.cb_records.find(cb_index);
    if (it == global.cb_records.end()) {
      return;
    }
    rec = it->second;
    global.cb_records.erase(it);
  }
  std::string line = kSchemaPrefix;
  line += "\"record\":\"cb\",\"command_buffer_index\":";
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
  write_line(global, std::move(line));
}

void note_wait(const char* bucket, uint64_t wait_ns) {
  ProducerGuard guard;
  if (!guard) {
    return;
  }
  auto& global = guard.state();
  const std::string name = bucket ? bucket : "";
  {
    std::lock_guard<std::mutex> lk(global.bucket_mtx);
    auto& b = global.buckets[name];
    b.count += 1;
    b.total_ns += wait_ns;
  }
  if (wait_ns < kWaitEmitFloorNs) {
    return;
  }
  std::string line = kSchemaPrefix;
  line += "\"record\":\"wait\",\"bucket\":";
  append_json_string(line, name);
  line += ",\"wait_ns\":" + std::to_string(wait_ns);
  line += ",\"at_ns\":" + std::to_string(now_ns());
  line += "}\n";
  write_line(global, std::move(line));
}

void flush() {
  ProducerGuard guard;
  if (!guard) {
    return;
  }
  auto& state = guard.state();
  const uint64_t summary_line = write_summary(state, false);
  wait_until_written(state, summary_line);
  std::lock_guard<std::mutex> lk(state.file_mtx);
  state.file.flush();
}

} // namespace mlx::core::census
