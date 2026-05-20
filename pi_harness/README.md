# Pi Harness

Python measurement scripts for Raspberry Pi 5 (4 GB). Run directly on the Pi via SSH.
Reports metrics as JSON conforming to `MetricsReport` schema back to the Compute Mac.

Built in Phase 0 Week 4 (Tasks 4.1–4.3). See `docs/decisions/0004-pi-measurement-methodology.md`
for measurement methodology decisions, including the pre-flight swap check.

## Structure

```
pi_harness/
  measure.py         — entry point; wraps llama.cpp / onnxruntime; emits MetricsReport JSON
  preflight.py       — checks free RAM and swap before each run; aborts if swap > 0
  report_sender.py   — posts MetricsReport JSON to Compute Mac via HTTP
  requirements.txt   — Pi-specific deps (minimal: psutil, requests)
```

## Usage

```bash
ssh pi@raspberrypi.local
cd ~/pi_harness
python3 measure.py --model models/lfm2-vl-450m-q4_0.gguf --device raspberry_pi_5_4gb
```

## Key constraint

Any run where `preflight.py` detects non-zero swap is aborted and logged as
`status: failed` with `error_message: "swap detected before run"`. Latency measurements
on a swapping Pi 5 are meaningless. See `HardwareFingerprint.swap_contaminated`.
