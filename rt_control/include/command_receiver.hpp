#pragma once

#include "command_queue.hpp"

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>

namespace dronn::rt {

class CommandReceiver {
public:
    CommandReceiver(
        std::string bind_host,
        uint16_t port,
        CommandQueue& queue,
        uint32_t stale_packet_ms = 2000);
    ~CommandReceiver();

    bool start();
    void stop();

    uint64_t packets_received() const { return packets_received_; }
    uint64_t packets_rejected() const { return packets_rejected_; }

private:
    bool open_socket();
    void recv_loop();

    std::string bind_host_;
    uint16_t port_;
    CommandQueue& queue_;
    uint32_t stale_packet_ms_;
    int socket_fd_{-1};
    std::thread thread_;
    std::atomic<bool> running_{false};
    std::atomic<uint64_t packets_received_{0};
    std::atomic<uint64_t packets_rejected_{0};
    uint32_t last_sequence_{0};
    bool have_sequence_{false};
};

}  // namespace dronn::rt
