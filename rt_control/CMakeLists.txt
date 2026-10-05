cmake_minimum_required(VERSION 3.16)
project(dronn_rt_control LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(dronn_rt_core
    src/command.cpp
    src/command_receiver.cpp
    src/control_loop.cpp
    src/flight_state_machine.cpp
    src/timing_metrics.cpp
    src/watchdog.cpp
)
target_include_directories(dronn_rt_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(dronn_rt_core PUBLIC pthread)

add_executable(dronn_rt_service src/main.cpp)
target_link_libraries(dronn_rt_service PRIVATE dronn_rt_core)

enable_testing()
add_executable(dronn_rt_tests tests/rt_tests.cpp)
target_link_libraries(dronn_rt_tests PRIVATE dronn_rt_core)
add_test(NAME dronn_rt_tests COMMAND dronn_rt_tests)
