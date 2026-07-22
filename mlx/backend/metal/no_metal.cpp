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
uint64_t CaptureReplay::replay_submit(const std::vector<array>&) {
  throw std::runtime_error("[CaptureReplay::replay_submit] No Metal back-end.");
}
void CaptureReplay::replay_wait(uint64_t) {
  throw std::runtime_error("[CaptureReplay::replay_wait] No Metal back-end.");
}
std::vector<array> CaptureReplay::read_outputs() {
  throw std::runtime_error("[CaptureReplay::read_outputs] No Metal back-end.");
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
