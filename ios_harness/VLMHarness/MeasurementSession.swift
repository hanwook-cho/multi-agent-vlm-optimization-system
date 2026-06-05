import Foundation

/// Holds configuration for one measurement run.
struct RunConfig {
    let modelKey: String       // e.g. "LFM2-VL-450M"
    let modelPath: String      // path to .gguf
    let mmprojPath: String     // path to mmproj .gguf
    let imagePaths: [String]   // 5-10 sample images
    let prompt: String
    let maxTokens: Int
    let nWarmup: Int           // warmup runs (results discarded)
    let nMeasure: Int          // measured runs
}

/// Statistics over multiple runs.
struct RunStats {
    let ttftMsValues: [Double]
    let tpsValues: [Double]
    let peakMemMBValues: [Double]

    var ttftMsMean: Double     { ttftMsValues.mean }
    var ttftMsStddev: Double   { ttftMsValues.stddev }
    var tpsMean: Double        { tpsValues.mean }
    var peakMemMBMean: Double  { peakMemMBValues.mean }
}

private extension Array where Element == Double {
    var mean: Double {
        guard !isEmpty else { return 0 }
        return reduce(0, +) / Double(count)
    }
    var stddev: Double {
        guard count > 1 else { return 0 }
        let m = mean
        return sqrt(map { ($0 - m) * ($0 - m) }.reduce(0, +) / Double(count - 1))
    }
}

/// Runs warmup + measured passes and aggregates stats.
@MainActor
class MeasurementSession: ObservableObject {
    @Published var log: [String] = []
    @Published var isRunning = false
    @Published var lastStats: RunStats?
    @Published var lastOutputSample: String = ""

    func run(config: RunConfig) async -> RunStats? {
        await MainActor.run { isRunning = true; log = [] }

        let task = Task<RunStats?, Never>.detached(priority: .userInitiated) { [weak self] in
            guard let self else { return nil }

            await self.appendLog("Loading model: \(config.modelKey)")
            let runner: LlamaVLMRunner
            do {
                runner = try LlamaVLMRunner(modelPath: config.modelPath,
                                            mmprojPath: config.mmprojPath)
            } catch {
                await self.appendLog("❌ Load failed: \(error.localizedDescription)")
                await MainActor.run { self.isRunning = false }
                return nil
            }
            await self.appendLog("✅ Model loaded. Warming up (\(config.nWarmup) run(s))…")

            // Warmup — discard results
            for i in 0 ..< config.nWarmup {
                let img = config.imagePaths.isEmpty ? nil : config.imagePaths[i % config.imagePaths.count]
                _ = try? runner.infer(withImagePath: img,
                                      prompt: config.prompt,
                                      maxTokens: config.maxTokens)
                await self.appendLog("  warmup \(i + 1) done")
            }

            await self.appendLog("Measuring (\(config.nMeasure) run(s))…")
            var ttftValues:    [Double] = []
            var tpsValues:     [Double] = []
            var memValues:     [Double] = []
            var outputSample = ""

            for i in 0 ..< config.nMeasure {
                let img = config.imagePaths.isEmpty ? nil : config.imagePaths[i % config.imagePaths.count]
                let result: VLMInferenceResult
                do {
                    result = try runner.infer(withImagePath: img,
                                             prompt: config.prompt,
                                             maxTokens: config.maxTokens)
                } catch {
                    await self.appendLog("  run \(i + 1) ❌ \(error.localizedDescription)")
                    continue
                }
                ttftValues.append(result.ttftMs)
                tpsValues.append(result.decodeTokensPerSec)
                memValues.append(result.peakMemoryMB)
                outputSample = result.text
                await self.appendLog(String(format: "  run %d: TTFT=%.1f ms  TPS=%.1f  mem=%.0f MB",
                                           i + 1, result.ttftMs, result.decodeTokensPerSec, result.peakMemoryMB))
            }

            guard !ttftValues.isEmpty else {
                await self.appendLog("❌ All measurement runs failed")
                await MainActor.run { self.isRunning = false }
                return nil
            }

            let stats = RunStats(ttftMsValues: ttftValues,
                                 tpsValues: tpsValues,
                                 peakMemMBValues: memValues)
            await self.appendLog(String(format: "✅ Done. TTFT=%.1f±%.1f ms  TPS=%.1f  mem=%.0f MB",
                                        stats.ttftMsMean, stats.ttftMsStddev,
                                        stats.tpsMean, stats.peakMemMBMean))
            await MainActor.run {
                self.lastStats = stats
                self.lastOutputSample = outputSample
                self.isRunning = false
            }
            return stats
        }
        return await task.value
    }

    private func appendLog(_ msg: String) async {
        await MainActor.run { log.append(msg) }
    }
}
