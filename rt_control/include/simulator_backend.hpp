#pragma once

#include "flight_backend.hpp"

#include <iostream>
#include <string>

namespace dronn::rt {

class SimulatorBackend final : public IFlightBackend {
public:
    bool connect() override {
        connected_ = true;
        airborne_ = false;
        std::cout << "[sim] connected\n";
        return true;
    }

    void disconnect() override {
        send_rc({0, 0, 0, 0});
        connected_ = false;
        airborne_ = false;
        std::cout << "[sim] disconnected\n";
    }

    bool takeoff() override {
        if (!connected_) {
            return false;
        }
        airborne_ = true;
        std::cout << "[sim] TAKEOFF\n";
        return true;
    }

    bool land() override {
        if (!connected_) {
            return false;
        }
        send_rc({0, 0, 0, 0});
        airborne_ = false;
        std::cout << "[sim] LAND\n";
        return true;
    }

    bool send_rc(const RcValues& rc) override {
        if (!connected_) {
            return false;
        }
        last_rc_ = rc;
        if (rc.lr || rc.fb || rc.ud || rc.yaw) {
            std::cout << "[sim] RC " << rc.lr << " " << rc.fb << " " << rc.ud << " "
                      << rc.yaw << "\n";
        }
        return true;
    }

    void emergency_stop() override {
        send_rc({0, 0, 0, 0});
        std::cout << "[sim] emergency_stop\n";
    }

    std::string name() const override { return "simulator"; }

    bool airborne() const { return airborne_; }

private:
    bool connected_{false};
    bool airborne_{false};
    RcValues last_rc_{};
};

}  // namespace dronn::rt
