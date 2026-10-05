#pragma once

#include "command_queue.hpp"
#include "command_receiver.hpp"
#include "flight_state_machine.hpp"
#include "flight_backend.hpp"
#include "simulator_backend.hpp"
#include "timing_metrics.hpp"
#include "watchdog.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

namespace dronn::rt {

struct ControlLoopConfig {
    double target_hz{20.0};
    uint32_t watchdog_hover_ms{500};
    uint32_t watchdog_safe_ms{2000};
    uint32_t stale_packet_ms{2000};
    std::string bind_host{"0.0.0.0"};
    uint16_t port{9999};
};

class ControlLoop {
public:
    explicit ControlLoop(ControlLoopConfig cfg);

    bool start();
    void stop();
    void run_until_signal();

    TimingMetrics metrics() const;

private:
    void tick(uint64_t now_ms);
    RcValues command_to_rc(const Command& cmd) const;
    void apply_backend_action(const Command& cmd, bool fsm_ok);
    uint64_t now_ms() const;

    ControlLoopConfig cfg_;
    CommandQueue queue_;
    std::unique_ptr<CommandReceiver> receiver_;
    std::unique_ptr<IFlightBackend> backend_;
    FlightStateMachine fsm_;
    Watchdog watchdog_;
    TimingMetrics metrics_;
    RcValues current_rc_{};
    std::atomic<bool> running_{false};
    bool watchdog_landed_{false};
    bool watchdog_hover_active_{false};
    bool watchdog_safe_active_{false};
};

}  // namespace dronn::rt
