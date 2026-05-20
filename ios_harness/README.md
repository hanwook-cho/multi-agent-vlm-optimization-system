# iOS Harness

Swift code for measuring inference on iPhone. Deployed as an iOS app to the
test device via Xcode. Reports metrics (TTFT, decode tokens/sec, peak memory,
on-disk size) as JSON conforming to `MetricsReport` schema back to the Compute Mac.

Built in Phase 0 Week 3 (Tasks 3.1–3.4). See `docs/decisions/0002-ios-measurement-methodology.md`
for measurement methodology decisions.

## Structure

```
ios_harness/
  VLMHarness.xcodeproj/
  VLMHarness/
    MeasurementSession.swift   — instruments timing + memory
    ModelRunner.swift          — thin wrapper around LEAP SDK / MLX / CoreML
    ReportExporter.swift       — serialises MetricsReport JSON, posts to Compute Mac
    ContentView.swift          — minimal SwiftUI UI (not the measurement logic)
```

## Requirements

- Xcode 15+, iOS 17+
- Apple Developer account ($99/year) with the test device registered
- See `docs/decisions/0001-ios-provisioning.md` (to be written in Phase 0 Week 3)
