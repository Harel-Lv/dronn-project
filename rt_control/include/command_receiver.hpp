#pragma once

#include "command_queue.hpp"

#include <netinet/in.h>

#include <atomic>
#include <cstdint>
#include <mutex>
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

    uint64_t packets_received() const { return packets_received_.load(); }
    uint64_t packets_rejected() const { return packets_rejected_.load(); }

    /** Reply to the source of the last accepted command (used for Connect ACK). */
    bool send_to_last_client(const uint8_t* data, size_t len);

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
    std::atomic<uint64_t> packets_received_{0};
    std::atomic<uint64_t> packets_rejected_{0};
    uint32_t last_sequence_{0};
    bool have_sequence_{false};
    std::mutex client_mutex_;
    sockaddr_in last_client_{};
    socklen_t last_client_len_{0};
    bool has_last_client_{false};
};

}  // namespace dronn::rt
