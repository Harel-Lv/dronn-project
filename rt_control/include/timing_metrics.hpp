#pragma once

#include <cstddef>
#include <cstdint>

namespace dronn::rt {

struct TimingMetrics {
    static constexpr std::size_t kExecRing = 64;

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
    double p95_exec_ms{0.0};
    double measured_hz{0.0};
    uint64_t rc_send_failures{0};
    double exec_ring_[kExecRing]{};
    std::size_t exec_ring_i_{0};
    std::size_t exec_ring_n_{0};

    void on_loop(double period_ms, double exec_ms, double jitter_ms, bool missed);
    void on_command_latency(double latency_ms);
    void maybe_print(uint64_t loop_count, double interval_sec);
    void print_summary() const;
};

}  // namespace dronn::rt
