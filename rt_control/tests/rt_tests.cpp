#include "command.hpp"
#include "flight_state_machine.hpp"
#include "watchdog.hpp"

#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>

using namespace dronn::rt;

static void test_clamp_rc() {
    assert(clamp_rc(-200) == -100);
    assert(clamp_rc(150) == 100);
    assert(clamp_rc(50) == 50);
}

static void test_decode_packet() {
    CommandPacket p{};
    p.version = kProtocolVersion;
    p.command = static_cast<uint8_t>(CommandType::RcDirect);
    p.sequence = 7;
    p.lr = 10;
    p.fb = -20;
    p.ud = 0;
    p.yaw = 5;
    Command c{};
    assert(decode_packet(p, c));
    assert(c.type == CommandType::RcDirect);
    assert(c.sequence == 7);
    assert(c.lr == 10 && c.fb == -20 && c.yaw == 5);
}

static void test_decode_bad_version() {
    CommandPacket p{};
    p.version = 99;
    p.command = static_cast<uint8_t>(CommandType::Hover);
    Command c{};
    assert(!decode_packet(p, c));
}

static void test_fsm_takeoff_land() {
    FlightStateMachine fsm;
    std::string reason;
    Command connect{};
    connect.type = CommandType::Connect;
    assert(fsm.apply_command(connect, reason));
    assert(fsm.state() == FlightState::Ready);

    Command to{};
    to.type = CommandType::Takeoff;
    assert(fsm.apply_command(to, reason));
    assert(fsm.state() == FlightState::TakingOff);
    fsm.on_takeoff_complete();
    assert(fsm.state() == FlightState::Hovering);

    Command land{};
    land.type = CommandType::Land;
    assert(fsm.apply_command(land, reason));
    assert(fsm.state() == FlightState::Landing);
    fsm.on_land_complete();
    assert(fsm.state() == FlightState::Ready);
}

static void test_watchdog() {
    Watchdog wd(100, 300);
    wd.on_valid_command(1000);
    assert(wd.evaluate(1050) == WatchdogAction::None);
    assert(wd.evaluate(1150) == WatchdogAction::Hover);
    assert(wd.evaluate(1350) == WatchdogAction::Safe);
}

static void test_fsm_rejects_motion_while_ready() {
    FlightStateMachine fsm;
    std::string reason;
    Command connect{};
    connect.type = CommandType::Connect;
    assert(fsm.apply_command(connect, reason));
    Command rc{};
    rc.type = CommandType::RcDirect;
    assert(!fsm.apply_command(rc, reason));
}

static void test_decode_clamps_rc_in_packet() {
    CommandPacket p{};
    p.version = kProtocolVersion;
    p.command = static_cast<uint8_t>(CommandType::RcDirect);
    p.lr = 200;
    p.fb = -150;
    Command c{};
    assert(decode_packet(p, c));
    assert(c.lr == 100 && c.fb == -100);
}

static void test_stale_timestamp() {
    assert(!is_packet_stale(1'000'000, 1'500'000, 1000));
    assert(is_packet_stale(1'000'000, 3'000'000, 1000));
}

static void test_future_timestamp() {
    assert(!is_packet_future(1'000'000, 1'000'000, 500));
    assert(!is_packet_future(1'400'000, 1'000'000, 500));
    assert(is_packet_future(2'000'000, 1'000'000, 500));
}

static void test_land_from_safe() {
    FlightStateMachine fsm;
    std::string reason;
    fsm.enter_safe();
    Command land{};
    land.type = CommandType::Land;
    assert(fsm.apply_command(land, reason));
    assert(fsm.state() == FlightState::Landing);
}

static void test_takeoff_failed_reverts_fsm() {
    FlightStateMachine fsm;
    std::string reason;
    Command connect{};
    connect.type = CommandType::Connect;
    assert(fsm.apply_command(connect, reason));
    Command to{};
    to.type = CommandType::Takeoff;
    assert(fsm.apply_command(to, reason));
    assert(fsm.state() == FlightState::TakingOff);
    fsm.on_takeoff_failed();
    assert(fsm.state() == FlightState::Ready);
}

static void test_sleep_until_cadence() {
    using namespace std::chrono;
    constexpr double hz = 100.0;
    const auto period = duration_cast<steady_clock::duration>(duration<double>(1.0 / hz));
    auto next = steady_clock::now() + period;
    double sum_ms = 0.0;
    for (int i = 0; i < 8; ++i) {
        const auto t0 = steady_clock::now();
        std::this_thread::sleep_until(next);
        const auto t1 = steady_clock::now();
        sum_ms += duration<double, std::milli>(t1 - t0).count();
        next += period;
    }
    const double avg = sum_ms / 8.0;
    assert(avg > 8.0 && avg < 14.0);
}

int main() {
    test_clamp_rc();
    test_decode_packet();
    test_decode_bad_version();
    test_fsm_takeoff_land();
    test_watchdog();
    test_fsm_rejects_motion_while_ready();
    test_decode_clamps_rc_in_packet();
    test_stale_timestamp();
    test_future_timestamp();
    test_land_from_safe();
    test_takeoff_failed_reverts_fsm();
    test_sleep_until_cadence();
    std::cout << "dronn_rt_tests: OK\n";
    return 0;
}
