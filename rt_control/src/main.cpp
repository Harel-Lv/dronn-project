#include "control_loop.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " [--bind HOST] [--port PORT] [--hz HZ]"
              << " [--watchdog-hover MS] [--watchdog-safe MS]\n";
}

}  // namespace

int main(int argc, char** argv) {
    dronn::rt::ControlLoopConfig cfg;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << "\n";
                usage(argv[0]);
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--bind") {
            cfg.bind_host = need("--bind");
        } else if (arg == "--port") {
            cfg.port = static_cast<uint16_t>(std::stoi(need("--port")));
        } else if (arg == "--hz") {
            cfg.target_hz = std::stod(need("--hz"));
        } else if (arg == "--watchdog-hover") {
            cfg.watchdog_hover_ms = static_cast<uint32_t>(std::stoul(need("--watchdog-hover")));
        } else if (arg == "--watchdog-safe") {
            cfg.watchdog_safe_ms = static_cast<uint32_t>(std::stoul(need("--watchdog-safe")));
        } else if (arg == "--stale-ms") {
            cfg.stale_packet_ms = static_cast<uint32_t>(std::stoul(need("--stale-ms")));
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown arg: " << arg << "\n";
            usage(argv[0]);
            return 2;
        }
    }

    if (cfg.target_hz <= 0.0) {
        std::cerr << "Invalid --hz (must be > 0)\n";
        return 2;
    }
    if (cfg.watchdog_safe_ms < cfg.watchdog_hover_ms) {
        std::cerr << "watchdog-safe must be >= watchdog-hover\n";
        return 2;
    }

    dronn::rt::ControlLoop loop(cfg);
    if (!loop.start()) {
        std::cerr << "Failed to start control loop\n";
        return 1;
    }
    loop.run_until_signal();
    return 0;
}
