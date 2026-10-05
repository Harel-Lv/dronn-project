#include "command_receiver.hpp"

#include "command.hpp"

#include <arpa/inet.h>
#include <chrono>
#include <cstring>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

namespace dronn::rt {

CommandReceiver::CommandReceiver(
    std::string bind_host,
    uint16_t port,
    CommandQueue& queue,
    uint32_t stale_packet_ms)
    : bind_host_(std::move(bind_host)),
      port_(port),
      queue_(queue),
      stale_packet_ms_(stale_packet_ms) {}

CommandReceiver::~CommandReceiver() { stop(); }

bool CommandReceiver::open_socket() {
    const int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) {
        std::cerr << "[receiver] socket failed\n";
        return false;
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port_);
    if (bind_host_ == "0.0.0.0" || bind_host_.empty()) {
        addr.sin_addr.s_addr = INADDR_ANY;
    } else if (::inet_pton(AF_INET, bind_host_.c_str(), &addr.sin_addr) != 1) {
        std::cerr << "[receiver] invalid bind host: " << bind_host_ << "\n";
        ::close(fd);
        return false;
    }

    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::cerr << "[receiver] bind failed on port " << port_ << "\n";
        ::close(fd);
        return false;
    }

    timeval rcv_timeout{};
    rcv_timeout.tv_sec = 0;
    rcv_timeout.tv_usec = 200000;
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &rcv_timeout, sizeof(rcv_timeout));

    socket_fd_ = fd;
    std::cout << "[receiver] listening UDP " << bind_host_ << ":" << port_ << "\n";
    return true;
}

bool CommandReceiver::start() {
    if (running_.load()) {
        return socket_fd_ >= 0;
    }
    if (!open_socket()) {
        return false;
    }
    running_ = true;
    thread_ = std::thread([this] { recv_loop(); });
    return true;
}

void CommandReceiver::stop() {
    if (!running_.exchange(false)) {
        if (socket_fd_ >= 0) {
            ::close(socket_fd_);
            socket_fd_ = -1;
        }
        return;
    }
    if (thread_.joinable()) {
        thread_.join();
    }
    if (socket_fd_ >= 0) {
        ::close(socket_fd_);
        socket_fd_ = -1;
    }
}

void CommandReceiver::recv_loop() {
    const int fd = socket_fd_;
    if (fd < 0) {
        running_ = false;
        return;
    }

    while (running_.load()) {
        CommandPacket packet{};
        sockaddr_in from{};
        socklen_t from_len = sizeof(from);
        const ssize_t n = ::recvfrom(
            fd, &packet, sizeof(packet), 0, reinterpret_cast<sockaddr*>(&from), &from_len);
        if (n < 0) {
            continue;
        }
        if (static_cast<std::size_t>(n) != kPacketSize) {
            packets_rejected_++;
            continue;
        }

        Command cmd{};
        if (!decode_packet(packet, cmd)) {
            packets_rejected_++;
            continue;
        }

        using namespace std::chrono;
        const uint64_t now_us = duration_cast<microseconds>(
                                    system_clock::now().time_since_epoch())
                                    .count();
        if (is_packet_stale(packet.timestamp_us, now_us, stale_packet_ms_)) {
            packets_rejected_++;
            continue;
        }
        if (is_packet_future(packet.timestamp_us, now_us, 500)) {
            packets_rejected_++;
            continue;
        }

        if (cmd.type == CommandType::Connect || cmd.type == CommandType::Disconnect) {
            have_sequence_ = false;
            last_sequence_ = 0;
        } else if (have_sequence_ && packet.sequence <= last_sequence_) {
            packets_rejected_++;
            continue;
        }

        have_sequence_ = true;
        last_sequence_ = packet.sequence;

        if (cmd.type == CommandType::Connect) {
            const uint8_t ack = kConnectAck;
            ::sendto(
                fd,
                &ack,
                1,
                0,
                reinterpret_cast<sockaddr*>(&from),
                from_len);
        }

        queue_.push(cmd);
        packets_received_++;
    }
}

}  // namespace dronn::rt
