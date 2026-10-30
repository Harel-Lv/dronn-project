#pragma once

#include "command.hpp"

#include <atomic>
#include <mutex>

namespace dronn::rt {

inline bool is_critical_command(CommandType type) {
    switch (type) {
        case CommandType::Takeoff:
        case CommandType::Land:
        case CommandType::Connect:
        case CommandType::Disconnect:
            return true;
        default:
            return false;
    }
}

inline int critical_command_priority(CommandType type) {
    switch (type) {
        case CommandType::Land:
        case CommandType::Disconnect:
            return 2;
        case CommandType::Takeoff:
        case CommandType::Connect:
            return 1;
        default:
            return 0;
    }
}

// Latest motion slot + one critical slot (LAND/TAKEOFF/CONNECT/DISCONNECT).
// Continuous RC updates overwrite only the motion slot so safety commands are not dropped.
// Land and Disconnect beat Takeoff/Connect in the critical slot.
class CommandQueue {
public:
    void push(const Command& cmd) {
        std::lock_guard<std::mutex> lock(mutex_);
        push_count_++;
        if (is_critical_command(cmd.type)) {
            if (!has_critical_.load(std::memory_order_relaxed) ||
                critical_command_priority(cmd.type) >=
                    critical_command_priority(critical_.type)) {
                critical_ = cmd;
                has_critical_.store(true, std::memory_order_release);
            }
            return;
        }
        latest_ = cmd;
        has_motion_.store(true, std::memory_order_release);
    }

    bool pop_latest(Command& out) {
        if (has_critical_.load(std::memory_order_acquire)) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (has_critical_.load(std::memory_order_relaxed)) {
                out = critical_;
                has_critical_.store(false, std::memory_order_release);
                return true;
            }
        }
        if (!has_motion_.load(std::memory_order_acquire)) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (!has_motion_.load(std::memory_order_relaxed)) {
            return false;
        }
        out = latest_;
        has_motion_.store(false, std::memory_order_release);
        return true;
    }

    const Command& peek_latest() const { return latest_; }

    bool has_command() const {
        return has_critical_.load(std::memory_order_acquire) ||
               has_motion_.load(std::memory_order_acquire);
    }

    uint64_t push_count() const { return push_count_; }

private:
    mutable std::mutex mutex_;
    Command latest_{};
    Command critical_{};
    std::atomic<bool> has_motion_{false};
    std::atomic<bool> has_critical_{false};
    std::atomic<uint64_t> push_count_{0};
};

}  // namespace dronn::rt
