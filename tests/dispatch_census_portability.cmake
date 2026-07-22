file(READ "${SOURCE}" CENSUS_SOURCE)

string(FIND "${CENSUS_SOURCE}"
            "#if defined(__APPLE__)\n#include <mach/mach_time.h>" MACH_GUARD)
string(FIND "${CENSUS_SOURCE}" "std::chrono::steady_clock" PORTABLE_CLOCK)

if(MACH_GUARD EQUAL -1 OR PORTABLE_CLOCK EQUAL -1)
  message(
    FATAL_ERROR
      "dispatch_census.cpp must guard Mach APIs and provide a portable clock")
endif()
