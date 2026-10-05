#pragma once

#include <cstdint>

namespace dronn::rt {

struct TimingMetrics {
    uint64_t loop_count{0};
    uint64_t deadline_misses{0};
    uint64_t commands_received{0};
    uint64_t packets_rejected{0};
    uint64_t fsm_rejected{0};
    double target_hz{20.0};
    double avg_period_ms{0.0};
    double max_period_ms{0.0};
    double avg_exec_ms{0.0};
    double max_exec_ms{0.0};
    double avg_jitter_ms{0.0};
    double max_jitter_ms{0.0};
    uint64_t watchdog_hover{0};
    uint64_t watchdog_safe{0};
    double avg_command_latency_ms{0.0};
    double max_command_latency_ms{0.0};
    uint64_t latency_samples{0};

    void on_loop(double period_ms, double exec_ms, double jitter_ms, bool missed);
    void on_command_latency(double latency_ms);
    void maybe_print(uint64_t loop_count, double interval_sec);
};

}  // namespace dronn::rt
