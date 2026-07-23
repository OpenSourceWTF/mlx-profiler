// Copyright © 2025 Apple Inc.

#include <stdexcept>

#include "mlx/backend/metal/metal.h"
#include "mlx/fast.h"

namespace mlx::core {

namespace metal {

bool is_available() {
  return false;
}

void start_capture(std::string) {}
void stop_capture() {}

const std::unordered_map<std::string, std::variant<std::string, size_t>>&
device_info() {
  throw std::runtime_error(
      "[metal::device_info] Cannot get device info without metal backend");
};

// --- S2b capture-replay: no-op / throwing stubs (no Metal backend) ---------
// Complete the opaque Impl so unique_ptr<Impl> is destructible in this TU.
struct CaptureReplay::Impl {};

void capture_begin() {
  throw std::runtime_error("[capture_begin] No Metal back-end.");
}

std::shared_ptr<CaptureReplay> CaptureReplay::capture(
    const std::vector<array>&,
    const std::vector<array>&) {
  throw std::runtime_error("[CaptureReplay::capture] No Metal back-end.");
}

CaptureReplay::CaptureReplay(std::unique_ptr<Impl>) {}
CaptureReplay::~CaptureReplay() = default;

std::vector<array> CaptureReplay::replay(const std::vector<array>&) {
  throw std::runtime_error("[CaptureReplay::replay] No Metal back-end.");
}
std::shared_ptr<CaptureReplay::FeedbackPlan> CaptureReplay::make_feedback_plan(
    const std::vector<std::pair<size_t, size_t>>&) const {
  throw std::runtime_error(
      "[CaptureReplay::make_feedback_plan] No Metal back-end.");
}
uint64_t CaptureReplay::replay_submit(
    const std::vector<array>&,
    const std::shared_ptr<FeedbackPlan>&) {
  throw std::runtime_error("[CaptureReplay::replay_submit] No Metal back-end.");
}
uint64_t CaptureReplay::replay_submit_partial(
    const std::vector<size_t>&,
    const std::vector<array>&,
    const std::shared_ptr<FeedbackPlan>&) {
  throw std::runtime_error(
      "[CaptureReplay::replay_submit_partial] No Metal back-end.");
}
void CaptureReplay::write_input_range(size_t, size_t, const array&) {
  throw std::runtime_error(
      "[CaptureReplay::write_input_range] No Metal back-end.");
}
void CaptureReplay::replay_wait(uint64_t) {
  throw std::runtime_error("[CaptureReplay::replay_wait] No Metal back-end.");
}
std::vector<array> CaptureReplay::read_outputs() {
  throw std::runtime_error("[CaptureReplay::read_outputs] No Metal back-end.");
}
array CaptureReplay::read_output(size_t) {
  throw std::runtime_error("[CaptureReplay::read_output] No Metal back-end.");
}
array CaptureReplay::read_input(size_t) {
  throw std::runtime_error("[CaptureReplay::read_input] No Metal back-end.");
}
CaptureReplay::LeafBufferInfo CaptureReplay::input_buffer_info(size_t) const {
  throw std::runtime_error(
      "[CaptureReplay::input_buffer_info] No Metal back-end.");
}
CaptureReplay::LeafBufferInfo CaptureReplay::output_buffer_info(size_t) const {
  throw std::runtime_error(
      "[CaptureReplay::output_buffer_info] No Metal back-end.");
}
std::vector<array> CaptureReplay::output_arrays(
    const std::vector<size_t>&) const {
  throw std::runtime_error("[CaptureReplay::output_arrays] No Metal back-end.");
}
bool CaptureReplay::residency_set_active() const {
  return false;
}
size_t CaptureReplay::num_commands() const {
  return 0;
}
size_t CaptureReplay::num_barriers() const {
  return 0;
}
size_t CaptureReplay::largest_barrier_free_run() const {
  return 0;
}
size_t CaptureReplay::num_raw_barriers() const {
  return 0;
}
size_t CaptureReplay::num_waw_barriers() const {
  return 0;
}
size_t CaptureReplay::max_buffer_write_count() const {
  return 0;
}
size_t CaptureReplay::hottest_barrier_buffer_barriers() const {
  return 0;
}
size_t CaptureReplay::renamable_writes() const {
  return 0;
}
size_t CaptureReplay::renamed_writes() const {
  return 0;
}
const std::vector<array>& CaptureReplay::inputs() const {
  static std::vector<array> empty;
  return empty;
}
const std::vector<array>& CaptureReplay::outputs() const {
  static std::vector<array> empty;
  return empty;
}
const std::vector<std::string>& CaptureReplay::command_kernel_names() const {
  static std::vector<std::string> empty;
  return empty;
}

// --- S4a chained submit: throwing stubs (no Metal backend) -----------------
std::shared_ptr<ChainPlan> make_chain_plan(
    const std::shared_ptr<CaptureReplay>&,
    size_t,
    const std::shared_ptr<CaptureReplay>&,
    size_t) {
  throw std::runtime_error("[make_chain_plan] No Metal back-end.");
}
uint64_t chain_submit(
    const std::vector<ChainStage>&,
    const std::vector<std::shared_ptr<ChainPlan>>&) {
  throw std::runtime_error("[chain_submit] No Metal back-end.");
}
std::vector<ChainStageSpanNs> chain_wait(uint64_t) {
  throw std::runtime_error("[chain_wait] No Metal back-end.");
}
// Pure arithmetic (no Metal state) — real impl even in the no-metal build so the
// conversion seam is testable everywhere (report §37.10).
double chain_ticks_to_ns(
    uint64_t tick_delta,
    double cpu_delta_ns,
    double gpu_delta_ticks,
    double fallback_ns_per_tick) {
  double ns_per_tick;
  if (gpu_delta_ticks > 0.0 && cpu_delta_ns > 0.0) {
    ns_per_tick = cpu_delta_ns / gpu_delta_ticks;
  } else if (gpu_delta_ticks == 0.0 && fallback_ns_per_tick > 0.0) {
    ns_per_tick = fallback_ns_per_tick;
  } else {
    return -1.0;
  }
  return static_cast<double>(tick_delta) * ns_per_tick;
}

// Pure string arithmetic (no Metal state) — real impl even in the no-metal build
// so the within-stage split parser seam is testable everywhere (report §40). Kept
// byte-for-byte identical to the metal-backend definition.
ChainSplitPlan chain_parse_split_spec(
    const std::string& spec,
    uint64_t num_commands) {
  ChainSplitPlan plan;
  auto colon = spec.find(':');
  if (colon == std::string::npos) {
    return plan;
  }
  const std::string head = spec.substr(0, colon);
  const std::string tail = spec.substr(colon + 1);
  if (head.empty()) {
    return plan;
  }
  uint64_t stage = 0;
  for (char c : head) {
    if (c < '0' || c > '9') {
      return plan;
    }
    stage = stage * 10 + static_cast<uint64_t>(c - '0');
    if (stage > 0xffffffffULL) {
      return plan;
    }
  }
  std::vector<uint64_t> idx;
  uint64_t cur = 0;
  bool any_digit = false;
  auto flush = [&]() -> bool {
    if (!any_digit) {
      return false;
    }
    idx.push_back(cur);
    cur = 0;
    any_digit = false;
    return true;
  };
  for (char c : tail) {
    if (c == ',') {
      if (!flush()) {
        return plan;
      }
    } else if (c >= '0' && c <= '9') {
      cur = cur * 10 + static_cast<uint64_t>(c - '0');
      any_digit = true;
      if (cur > 0xffffffffULL) {
        return plan;
      }
    } else {
      return plan;
    }
  }
  if (!flush()) {
    return plan;
  }
  if (idx.empty() || num_commands == 0) {
    return plan;
  }
  uint64_t prev = 0;
  for (uint64_t v : idx) {
    if (v <= prev || v >= num_commands) {
      return plan;
    }
    prev = v;
  }
  plan.groups.reserve(idx.size() + 1);
  uint64_t start = 0;
  for (uint64_t v : idx) {
    plan.groups.emplace_back(start, v - start);
    start = v;
  }
  plan.groups.emplace_back(start, num_commands - start);
  plan.valid = true;
  plan.stage_index = static_cast<uint32_t>(stage);
  return plan;
}

} // namespace metal

namespace fast {

CustomKernelFunction metal_kernel(
    const std::string&,
    const std::vector<std::string>&,
    const std::vector<std::string>&,
    const std::string&,
    const std::string&,
    bool,
    bool) {
  throw std::runtime_error("[metal_kernel] No Metal back-end.");
}

} // namespace fast

} // namespace mlx::core
