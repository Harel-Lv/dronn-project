#include "control_loop.hpp"

#include <chrono>
#include <cmath>
#include <csignal>
#include <iostream>
#include <thread>

namespace dronn::rt {

namespace {
std::atomic<bool> g_stop{false};

void on_signal(int) { g_stop = true; }
}  // namespace

ControlLoop::ControlLoop(ControlLoopConfig cfg)
    : cfg_(std::move(cfg)),
      watchdog_(cfg_.watchdog_hover_ms, cfg_.watchdog_safe_ms) {
    metrics_.target_hz = cfg_.target_hz;
    backend_ = std::make_unique<SimulatorBackend>();
}

bool ControlLoop::start() {
    receiver_ = std::make_unique<CommandReceiver>(
        cfg_.bind_host, cfg_.port, queue_, cfg_.stale_packet_ms);
    if (!receiver_->start()) {
        return false;
    }
    running_ = true;
    std::cout << "[control] backend=" << backend_->name() << " hz=" << cfg_.target_hz << "\n";
    return true;
}

void ControlLoop::stop() {
    running_ = false;
    if (receiver_) {
        receiver_->stop();
    }
    if (backend_) {
        backend_->send_rc({0, 0, 0, 0});
        backend_->disconnect();
    }
}

TimingMetrics ControlLoop::metrics() const { return metrics_; }

uint64_t ControlLoop::now_ms() const {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

RcValues ControlLoop::command_to_rc(const Command& cmd) const {
    RcValues rc{};
    switch (cmd.type) {
        case CommandType::Hover:
            return rc;
        case CommandType::RcDirect:
            rc.lr = cmd.lr;
            rc.fb = cmd.fb;
            rc.ud = cmd.ud;
            rc.yaw = cmd.yaw;
            return rc;
        case CommandType::Forward:
            rc.fb = cmd.fb != 0 ? cmd.fb : 40;
            return rc;
        case CommandType::Back:
            rc.fb = cmd.fb != 0 ? cmd.fb : -40;
            return rc;
        case CommandType::Left:
            rc.lr = cmd.lr != 0 ? cmd.lr : -40;
            return rc;
        case CommandType::Right:
            rc.lr = cmd.lr != 0 ? cmd.lr : 40;
            return rc;
        case CommandType::Up:
            rc.ud = cmd.ud != 0 ? cmd.ud : 40;
            return rc;
        case CommandType::Down:
            rc.ud = cmd.ud != 0 ? cmd.ud : -40;
            return rc;
        case CommandType::RotateCw:
            rc.yaw = cmd.yaw != 0 ? cmd.yaw : 40;
            return rc;
        case CommandType::RotateCcw:
            rc.yaw = cmd.yaw != 0 ? cmd.yaw : -40;
            return rc;
        default:
            return current_rc_;
    }
}

void ControlLoop::send_command_ack(const Command& cmd, bool ok) {
    if (receiver_ == nullptr) {
        return;
    }
    receiver_->send_command_ack(
        ok ? 0 : 1,
        flight_state_to_u8(fsm_.state()),
        cmd.sequence);
}

void ControlLoop::apply_backend_action(const Command& cmd, bool fsm_ok) {
    if (!fsm_ok) {
        return;
    }
    switch (cmd.type) {
        case CommandType::Connect: {
            const bool ok = backend_->connect();
            if (ok) {
                fsm_.on_backend_connected();
                queue_.clear_motion();
            } else {
                std::cerr << "[control] backend connect failed\n";
            }
            send_command_ack(cmd, ok);
            break;
        }
        case CommandType::Disconnect:
            backend_->send_rc({0, 0, 0, 0});
            backend_->disconnect();
            queue_.clear_all();
            watchdog_landed_ = false;
            send_command_ack(cmd, true);
            break;
        case CommandType::Recover: {
            const bool grounded = backend_ != nullptr && !backend_->is_airborne();
            if (grounded) {
                fsm_.clear_safety_latch();
                queue_.clear_motion();
            }
            send_command_ack(cmd, grounded);
            if (!grounded) {
                std::cerr << "[control] recover rejected — backend not grounded\n";
            }
            break;
        }
        case CommandType::Takeoff:
            if (fsm_.state() == FlightState::TakingOff) {
                backend_->connect();
                const bool ok = backend_->takeoff();
                if (ok) {
                    fsm_.on_takeoff_complete();
                    queue_.clear_motion();
                } else {
                    fsm_.on_takeoff_failed();
                    std::cerr << "[control] takeoff failed (backend not ready?)\n";
                }
                send_command_ack(cmd, ok);
            }
            break;
        case CommandType::Land:
            if (fsm_.state() == FlightState::Landing) {
                const bool ok = backend_->land();
                if (ok) {
                    fsm_.on_land_complete();
                    queue_.clear_motion();
                    watchdog_landed_ = false;
                } else {
                    fsm_.on_land_failed();
                    std::cerr << "[control] land failed (backend not ready?)\n";
                }
                send_command_ack(cmd, ok);
            } else if (fsm_.state() == FlightState::Ready) {
                queue_.clear_motion();
                send_command_ack(cmd, true);
            }
            break;
        case CommandType::Hover:
        case CommandType::Forward:
        case CommandType::Back:
        case CommandType::Left:
        case CommandType::Right:
        case CommandType::Up:
        case CommandType::Down:
        case CommandType::RotateCw:
        case CommandType::RotateCcw:
        case CommandType::RcDirect:
            current_rc_ = command_to_rc(cmd);
            if (!backend_->send_rc(current_rc_)) {
                metrics_.rc_send_failures++;
                std::cerr << "[control] send_rc rejected by backend\n";
            }
            break;
        default:
            break;
    }
}

void ControlLoop::tick(uint64_t now_ms) {
    Command cmd{};
    if (queue_.pop_latest(cmd)) {
        std::string reject;
        const bool ok = fsm_.apply_command(cmd, reject);
        if (!ok) {
            metrics_.fsm_rejected++;
            std::cerr << "[control] reject seq=" << cmd.sequence << " " << reject << "\n";
            if (is_critical_command(cmd.type)) {
                send_command_ack(cmd, false);
            }
        } else {
            using namespace std::chrono;
            const uint64_t now_us = duration_cast<microseconds>(
                                        system_clock::now().time_since_epoch())
                                        .count();
            if (cmd.timestamp_us > 0 && now_us >= cmd.timestamp_us) {
                const double lat_ms =
                    static_cast<double>(now_us - cmd.timestamp_us) / 1000.0;
                metrics_.on_command_latency(lat_ms);
            }
            watchdog_.on_valid_command(now_ms);
            watchdog_hover_active_ = false;
            watchdog_safe_active_ = false;
            apply_backend_action(cmd, true);
        }
    }

    const WatchdogAction wd = watchdog_.evaluate(now_ms);
    if (wd == WatchdogAction::Hover) {
        if (!watchdog_hover_active_) {
            watchdog_hover_active_ = true;
            current_rc_ = {};
            backend_->send_rc(current_rc_);
            watchdog_.record_hover_activation();
            metrics_.watchdog_hover++;
        }
    } else {
        watchdog_hover_active_ = false;
    }
    if (wd == WatchdogAction::Safe) {
        if (!watchdog_safe_active_) {
            watchdog_safe_active_ = true;
            backend_->emergency_stop();
            current_rc_ = {};
            queue_.clear_motion();
            fsm_.enter_safe();
            watchdog_land_attempts_ = 0;
            last_watchdog_land_attempt_ms_ = 0;
            watchdog_.record_safe_activation();
            metrics_.watchdog_safe++;
        }
        if (!watchdog_landed_ &&
            watchdog_land_attempts_ < kMaxWatchdogLandAttempts &&
            (watchdog_land_attempts_ == 0 ||
             now_ms >= last_watchdog_land_attempt_ms_ + kWatchdogLandRetryMs)) {
            last_watchdog_land_attempt_ms_ = now_ms;
            watchdog_land_attempts_++;
            if (backend_->land()) {
                fsm_.on_land_complete();
                queue_.clear_all();
                watchdog_landed_ = true;
            } else {
                fsm_.on_land_failed();
                std::cerr << "[control] watchdog controlled land failed (attempt "
                          << static_cast<unsigned>(watchdog_land_attempts_) << "/"
                          << static_cast<unsigned>(kMaxWatchdogLandAttempts)
                          << ") — operator must recover\n";
            }
        }
    } else {
        watchdog_safe_active_ = false;
        if (wd != WatchdogAction::Safe) {
            watchdog_land_attempts_ = 0;
        }
    }

    if (receiver_) {
        metrics_.commands_received = receiver_->packets_received();
        metrics_.packets_rejected = receiver_->packets_rejected();
    }
}

void ControlLoop::run_until_signal() {
    using namespace std::chrono;
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    const double period_sec = 1.0 / cfg_.target_hz;
    const auto period = duration_cast<steady_clock::duration>(
        duration<double>(period_sec));
    auto next = steady_clock::now() + period;
    auto prev_wake = steady_clock::now();

    while (running_.load() && !g_stop.load()) {
        const auto wake = steady_clock::now();
        const double period_ms =
            duration_cast<duration<double, std::milli>>(wake - prev_wake).count();
        prev_wake = wake;

        const auto exec_start = steady_clock::now();
        tick(now_ms());
        const auto exec_end = steady_clock::now();
        const double exec_ms =
            duration_cast<duration<double, std::milli>>(exec_end - exec_start).count();

        const double jitter_ms =
            duration_cast<duration<double, std::milli>>(exec_end - next).count();
        const bool missed = exec_end > next;
        metrics_.on_loop(period_ms, exec_ms, jitter_ms, missed);
        metrics_.maybe_print(metrics_.loop_count, 5.0);

        std::this_thread::sleep_until(next);
        next += period;
    }

    stop();
    metrics_.print_summary();
}

}  // namespace dronn::rt
