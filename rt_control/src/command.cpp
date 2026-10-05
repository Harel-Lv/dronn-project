#include "command.hpp"

namespace dronn::rt {

int clamp_rc(int value, int lo, int hi) {
    if (value < lo) {
        return lo;
    }
    if (value > hi) {
        return hi;
    }
    return value;
}

static CommandType from_u8(uint8_t v) {
    if (v > static_cast<uint8_t>(CommandType::Disconnect)) {
        return CommandType::None;
    }
    return static_cast<CommandType>(v);
}

bool decode_packet(const CommandPacket& packet, Command& out) {
    out = Command{};
    if (packet.version != kProtocolVersion) {
        return false;
    }
    out.type = from_u8(packet.command);
    if (out.type == CommandType::None) {
        return false;
    }
    out.sequence = packet.sequence;
    out.timestamp_us = packet.timestamp_us;
    out.lr = static_cast<int16_t>(clamp_rc(packet.lr));
    out.fb = static_cast<int16_t>(clamp_rc(packet.fb));
    out.ud = static_cast<int16_t>(clamp_rc(packet.ud));
    out.yaw = static_cast<int16_t>(clamp_rc(packet.yaw));
    out.valid = true;
    return true;
}

bool is_packet_stale(uint64_t timestamp_us, uint64_t now_us, uint32_t max_age_ms) {
    if (timestamp_us == 0) {
        return false;
    }
    if (now_us < timestamp_us) {
        return false;
    }
    const uint64_t age_us = now_us - timestamp_us;
    const uint64_t max_us = static_cast<uint64_t>(max_age_ms) * 1000ULL;
    return age_us > max_us;
}

bool is_packet_future(uint64_t timestamp_us, uint64_t now_us, uint32_t max_future_ms) {
    if (timestamp_us == 0 || timestamp_us <= now_us) {
        return false;
    }
    const uint64_t ahead_us = timestamp_us - now_us;
    const uint64_t max_us = static_cast<uint64_t>(max_future_ms) * 1000ULL;
    return ahead_us > max_us;
}

}  // namespace dronn::rt
