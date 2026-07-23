#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "mlx/backend/common/dispatch_census.h"

namespace census = mlx::core::census;

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: dispatch_census_queue_smoke <jsonl>\n";
    return 1;
  }
  const std::filesystem::path path = argv[1];
  std::filesystem::remove(path);

  census::detail::set_writer_paused_for_test(true);
  census::initialize();
  for (int i = 0; i < 100; ++i) {
    census::note_wait("bounded_queue_probe", 1000);
  }
  census::detail::set_writer_paused_for_test(false);
  census::flush();

  std::ifstream input(path);
  std::ostringstream buffer;
  buffer << input.rdbuf();
  const std::string jsonl = buffer.str();
  if (jsonl.find("\"dropped_rows\":96") == std::string::npos ||
      jsonl.find("\"complete\":false") == std::string::npos) {
    std::cerr << "bounded queue did not report dropped rows\n";
    return 1;
  }
  return 0;
}
