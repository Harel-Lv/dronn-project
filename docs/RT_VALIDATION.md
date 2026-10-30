# Validating `rt_control` (Phase B)

Requires **Linux or WSL** (POSIX UDP). Native Windows builds are not supported for the service.

```bash
cd rt_control
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/dronn_rt_tests
```

From repo root:

```bash
bash scripts/build_rt_control.sh
python scripts/rt_cpp_ipc_demo.py   # service must be running
```

On this developer machine, WSL was not installed when Phase B landed — run the commands above in WSL/Linux and confirm `dronn_rt_tests: OK` before claiming C++ validation in release notes.
