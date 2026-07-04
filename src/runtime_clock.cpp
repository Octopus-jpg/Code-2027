#include "htc/runtime_clock.hpp"

#include <chrono>

namespace htc {
namespace {

using SteadyClock = std::chrono::steady_clock;

SteadyClock::time_point& runtime_start_time() {
    static SteadyClock::time_point start = SteadyClock::now();
    return start;
}

}  // namespace

void reset_runtime_clock() {
    runtime_start_time() = SteadyClock::now();
}

double runtime_elapsed_seconds() {
    return std::chrono::duration<double>(SteadyClock::now() - runtime_start_time()).count();
}

}  // namespace htc
