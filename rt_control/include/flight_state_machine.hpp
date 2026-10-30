#pragma once

#include "command.hpp"

#include <string>

namespace dronn::rt {

enum class FlightState {
    Disconnected,
    Ready,
    TakingOff,
    Flying,
    Hovering,
    Landing,
    Safe,
};

uint8_t flight_state_to_u8(FlightState s);

class FlightStateMachine {
public:
    FlightState state() const { return state_; }
    bool safety_latched() const { return safety_latched_; }

    static const char* state_name(FlightState s);

    bool can_accept_motion() const;
    bool apply_command(const Command& cmd, std::string& reject_reason);

    void on_backend_connected();
    void on_backend_disconnected();
    void on_takeoff_complete();
    void on_takeoff_failed();
    void on_land_complete();
    void on_land_failed();
    void enter_safe();
    void clear_safety_latch();

private:
    bool transition(FlightState next);
    bool handle_takeoff(std::string& reason);
    bool handle_land(std::string& reason);
    bool handle_motion_command(CommandType type, std::string& reason);
    bool rejects_due_to_safety_lock(CommandType type) const;

    FlightState state_{FlightState::Disconnected};
    bool safety_latched_{false};
};

}  // namespace dronn::rt
