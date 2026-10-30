#include "timing_metrics.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace dronn::rt {

namespace {
double percentile(std::vector<double>& v, double p) {
    if (v.empty()) {
        return 0.0;
    }
    const std::size_t idx =
        static_cast<std::size_t>(std::ceil(p * static_cast<double>(v.size()))) - 1;
    std::nth_element(v.begin(), v.begin() + idx, v.end());
    return v[idx];
}
}  // namespace

void TimingMetrics::on_loop(double period_ms, double exec_ms, double jitter_ms, bool missed) {
    loop_count++;
    if (period_ms > 0.01) {
        measured_hz = 1000.0 / period_ms;
    }
    exec_ring_[exec_ring_i_] = exec_ms;
    exec_ring_i_ = (exec_ring_i_ + 1) % kExecRing;
    if (exec_ring_n_ < kExecRing) {
        exec_ring_n_++;
    }
    if (missed) {
        deadline_misses++;
    }
    avg_period_ms += (period_ms - avg_period_ms) / static_cast<double>(loop_count);
    avg_exec_ms += (exec_ms - avg_exec_ms) / static_cast<double>(loop_count);
    avg_jitter_ms += (jitter_ms - avg_jitter_ms) / static_cast<double>(loop_count);
    if (period_ms > max_period_ms) {
        max_period_ms = period_ms;
    }
    if (exec_ms > max_exec_ms) {
        max_exec_ms = exec_ms;
    }
    if (jitter_ms > max_jitter_ms) {
        max_jitter_ms = jitter_ms;
    }
}

void TimingMetrics::on_command_latency(double latency_ms) {
    latency_samples++;
    avg_command_latency_ms +=
        (latency_ms - avg_command_latency_ms) / static_cast<double>(latency_samples);
    if (latency_ms > max_command_latency_ms) {
        max_command_latency_ms = latency_ms;
    }
}

void TimingMetrics::maybe_print(uint64_t count, double interval_sec) {
    if (count == 0 || count % static_cast<uint64_t>(target_hz * interval_sec) != 0) {
        return;
    }
    std::cout << "[metrics] target_hz=" << target_hz << " avg_period_ms=" << avg_period_ms
              << " max_period_ms=" << max_period_ms << " avg_exec_ms=" << avg_exec_ms
              << " max_exec_ms=" << max_exec_ms << " avg_jitter_ms=" << avg_jitter_ms
              << " max_jitter_ms=" << max_jitter_ms << " deadline_misses=" << deadline_misses
              << " rx=" << commands_received << " pkt_rejected=" << packets_rejected
              << " fsm_rejected=" << fsm_rejected
              << " wd_hover=" << watchdog_hover << " wd_safe=" << watchdog_safe
              << " avg_cmd_latency_ms=" << avg_command_latency_ms
              << " max_cmd_latency_ms=" << max_command_latency_ms << "\n";
}

}  // namespace dronn::rt
