#pragma once

#include "command.hpp"

#include <atomic>
#include <mutex>

namespace dronn::rt {

// Latest-command slot: receiver writes, control loop reads (no unbounded queue).
class CommandQueue {
public:
    void push(const Command& cmd) {
        std::lock_guard<std::mutex> lock(mutex_);
        latest_ = cmd;
        has_new_.store(true, std::memory_order_release);
        push_count_++;
    }

    bool pop_latest(Command& out) {
        if (!has_new_.load(std::memory_order_acquire)) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (!has_new_.load(std::memory_order_relaxed)) {
            return false;
        }
        out = latest_;
        has_new_.store(false, std::memory_order_release);
        return true;
    }

    const Command& peek_latest() const { return latest_; }

    bool has_command() const { return has_new_.load(std::memory_order_acquire); }

    uint64_t push_count() const { return push_count_; }

private:
    mutable std::mutex mutex_;
    Command latest_{};
    std::atomic<bool> has_new_{false};
    std::atomic<uint64_t> push_count_{0};
};

}  // namespace dronn::rt
