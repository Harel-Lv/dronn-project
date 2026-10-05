#include "watchdog.hpp"

namespace dronn::rt {

Watchdog::Watchdog(uint32_t hover_ms, uint32_t safe_ms)
    : hover_ms_(hover_ms), safe_ms_(safe_ms) {}

void Watchdog::on_valid_command(uint64_t now_ms) {
    last_command_ms_ = now_ms;
}

WatchdogAction Watchdog::evaluate(uint64_t now_ms) const {
    if (last_command_ms_ == 0) {
        return WatchdogAction::None;
    }
    const uint64_t elapsed = now_ms - last_command_ms_;
    if (elapsed >= safe_ms_) {
        return WatchdogAction::Safe;
    }
    if (elapsed >= hover_ms_) {
        return WatchdogAction::Hover;
    }
    return WatchdogAction::None;
}

}  // namespace dronn::rt
