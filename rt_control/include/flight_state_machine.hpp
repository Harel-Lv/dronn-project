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

class FlightStateMachine {
public:
    FlightState state() const { return state_; }

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

private:
    bool transition(FlightState next);
    bool handle_takeoff(std::string& reason);
    bool handle_land(std::string& reason);
    bool handle_motion_command(CommandType type, std::string& reason);

    FlightState state_{FlightState::Disconnected};
};

}  // namespace dronn::rt
