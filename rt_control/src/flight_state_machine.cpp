#include "flight_state_machine.hpp"

namespace dronn::rt {

uint8_t flight_state_to_u8(FlightState s) {
    return static_cast<uint8_t>(s);
}

const char* FlightStateMachine::state_name(FlightState s) {
    switch (s) {
        case FlightState::Disconnected:
            return "DISCONNECTED";
        case FlightState::Ready:
            return "READY";
        case FlightState::TakingOff:
            return "TAKING_OFF";
        case FlightState::Flying:
            return "FLYING";
        case FlightState::Hovering:
            return "HOVERING";
        case FlightState::Landing:
            return "LANDING";
        case FlightState::Safe:
            return "SAFE";
    }
    return "UNKNOWN";
}

bool FlightStateMachine::can_accept_motion() const {
    return state_ == FlightState::Flying || state_ == FlightState::Hovering;
}

void FlightStateMachine::on_backend_connected() {
    transition(FlightState::Ready);
}

void FlightStateMachine::on_backend_disconnected() {
    state_ = FlightState::Disconnected;
}

void FlightStateMachine::on_takeoff_complete() {
    if (state_ == FlightState::TakingOff) {
        transition(FlightState::Hovering);
    }
}

void FlightStateMachine::on_takeoff_failed() {
    if (state_ == FlightState::TakingOff) {
        transition(FlightState::Ready);
    }
}

void FlightStateMachine::on_land_complete() {
    if (state_ == FlightState::Landing || state_ == FlightState::Safe) {
        transition(FlightState::Ready);
    }
}

void FlightStateMachine::on_land_failed() {
    if (state_ == FlightState::Landing) {
        transition(safety_latched_ ? FlightState::Safe : FlightState::Hovering);
    }
    // If land fails while already in Safe (watchdog), remain latched in Safe.
}

void FlightStateMachine::enter_safe() {
    safety_latched_ = true;
    transition(FlightState::Safe);
}

void FlightStateMachine::clear_safety_latch() {
    safety_latched_ = false;
    if (state_ == FlightState::Safe) {
        transition(FlightState::Ready);
    }
}

bool FlightStateMachine::rejects_due_to_safety_lock(CommandType type) const {
    if (!safety_latched_) {
        return false;
    }
    switch (type) {
        case CommandType::Takeoff:
        case CommandType::Forward:
        case CommandType::Back:
        case CommandType::Left:
        case CommandType::Right:
        case CommandType::Up:
        case CommandType::Down:
        case CommandType::RotateCw:
        case CommandType::RotateCcw:
        case CommandType::RcDirect:
            return true;
        default:
            return false;
    }
}

bool FlightStateMachine::transition(FlightState next) {
    state_ = next;
    return true;
}

bool FlightStateMachine::handle_takeoff(std::string& reason) {
    if (safety_latched_) {
        reason = "safety lock active";
        return false;
    }
    if (state_ == FlightState::TakingOff || state_ == FlightState::Flying ||
        state_ == FlightState::Hovering || state_ == FlightState::Landing) {
        reason = "already airborne or takeoff in progress";
        return false;
    }
    if (state_ == FlightState::Ready) {
        transition(FlightState::TakingOff);
        return true;
    }
    reason = "takeoff not allowed in current state";
    return false;
}

bool FlightStateMachine::handle_land(std::string& reason) {
    if (state_ == FlightState::Flying || state_ == FlightState::Hovering ||
        state_ == FlightState::TakingOff || state_ == FlightState::Safe) {
        transition(FlightState::Landing);
        return true;
    }
    if (state_ == FlightState::Ready) {
        reason = "already on ground";
        return true;
    }
    reason = "land not allowed in current state";
    return false;
}

bool FlightStateMachine::handle_motion_command(CommandType type, std::string& reason) {
    if (type == CommandType::Hover) {
        if (can_accept_motion() || state_ == FlightState::TakingOff) {
            transition(FlightState::Hovering);
            return true;
        }
        if (state_ == FlightState::Ready) {
            return true;
        }
        reason = "hover rejected";
        return false;
    }
    if (!can_accept_motion()) {
        reason = "motion while not flying";
        return false;
    }
    transition(FlightState::Flying);
    (void)type;
    return true;
}

bool FlightStateMachine::apply_command(const Command& cmd, std::string& reject_reason) {
    if (rejects_due_to_safety_lock(cmd.type)) {
        reject_reason = "safety lock active";
        return false;
    }
    switch (cmd.type) {
        case CommandType::Recover:
            if (!safety_latched_) {
                reject_reason = "not in safety lock";
                return false;
            }
            return true;
        case CommandType::Connect:
            if (state_ != FlightState::Disconnected) {
                reject_reason = "already connected";
                return false;
            }
            return true;
        case CommandType::Disconnect:
            if (state_ == FlightState::Disconnected) {
                return true;
            }
            transition(FlightState::Disconnected);
            return true;
        case CommandType::Takeoff:
            return handle_takeoff(reject_reason);
        case CommandType::Land:
            return handle_land(reject_reason);
        case CommandType::Hover:
        case CommandType::Forward:
        case CommandType::Back:
        case CommandType::Left:
        case CommandType::Right:
        case CommandType::Up:
        case CommandType::Down:
        case CommandType::RotateCw:
        case CommandType::RotateCcw:
        case CommandType::RcDirect:
            return handle_motion_command(cmd.type, reject_reason);
        default:
            reject_reason = "unknown command";
            return false;
    }
}

}  // namespace dronn::rt
