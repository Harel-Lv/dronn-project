#pragma once

#include <cstdint>

namespace dronn::rt {

enum class WatchdogAction {
    None,
    Hover,
    Safe,
};

class Watchdog {
public:
    explicit Watchdog(uint32_t hover_ms, uint32_t safe_ms);

    void on_valid_command(uint64_t now_ms);
    WatchdogAction evaluate(uint64_t now_ms) const;

    uint64_t activations_hover() const { return activations_hover_; }
    uint64_t activations_safe() const { return activations_safe_; }

    void record_hover_activation() { activations_hover_++; }
    void record_safe_activation() { activations_safe_++; }

private:
    uint32_t hover_ms_;
    uint32_t safe_ms_;
    uint64_t last_command_ms_{0};
    mutable uint64_t activations_hover_{0};
    mutable uint64_t activations_safe_{0};
};

}  // namespace dronn::rt
