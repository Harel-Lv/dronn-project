#pragma once

#include <cstdint>

namespace dronn::rt {

constexpr uint8_t kProtocolVersion = 1;
constexpr std::size_t kPacketSize = 24;
constexpr uint8_t kConnectAck = 0xAC;

enum class CommandType : uint8_t {
    None = 0,
    Takeoff = 1,
    Land = 2,
    Hover = 3,
    Forward = 4,
    Back = 5,
    Left = 6,
    Right = 7,
    Up = 8,
    Down = 9,
    RotateCw = 10,
    RotateCcw = 11,
    RcDirect = 12,
    Connect = 13,
    Disconnect = 14,
};

#pragma pack(push, 1)
struct CommandPacket {
    uint8_t version{kProtocolVersion};
    uint8_t command{0};
    uint16_t reserved{0};
    uint32_t sequence{0};
    uint64_t timestamp_us{0};
    int16_t lr{0};
    int16_t fb{0};
    int16_t ud{0};
    int16_t yaw{0};
};
#pragma pack(pop)

static_assert(sizeof(CommandPacket) == kPacketSize, "CommandPacket size mismatch");

struct Command {
    CommandType type{CommandType::None};
    uint32_t sequence{0};
    uint64_t timestamp_us{0};
    int16_t lr{0};
    int16_t fb{0};
    int16_t ud{0};
    int16_t yaw{0};
    bool valid{false};
};

int clamp_rc(int value, int lo = -100, int hi = 100);

bool decode_packet(const CommandPacket& packet, Command& out);

// Wall-clock age check (timestamp_us from Python time.time()).
bool is_packet_stale(uint64_t timestamp_us, uint64_t now_us, uint32_t max_age_ms);

bool is_packet_future(uint64_t timestamp_us, uint64_t now_us, uint32_t max_future_ms);

}  // namespace dronn::rt
