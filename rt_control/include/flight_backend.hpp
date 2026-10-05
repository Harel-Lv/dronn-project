#pragma once

#include <cstdint>
#include <string>

namespace dronn::rt {

struct RcValues {
    int lr{0};
    int fb{0};
    int ud{0};
    int yaw{0};
};

class IFlightBackend {
public:
    virtual ~IFlightBackend() = default;
    virtual bool connect() = 0;
    virtual void disconnect() = 0;
    virtual bool takeoff() = 0;
    virtual bool land() = 0;
    virtual bool send_rc(const RcValues& rc) = 0;
    virtual void emergency_stop() = 0;
    virtual std::string name() const = 0;
};

}  // namespace dronn::rt
