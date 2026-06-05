import Foundation

/// Serialises a measurement run into a MetricsReport JSON compatible
/// with the project's schemas/MetricsReport schema.
struct ReportExporter {

    struct ModelInfo {
        let key: String       // e.g. "LFM2-VL-450M"
        let hfId: String      // e.g. "LiquidAI/LFM2-VL-450M"
        let quantization: String // e.g. "Q4_0"
        let onDiskSizeMB: Double // GGUF file size
    }

    static func export(
        modelInfo: ModelInfo,
        stats: RunStats,
        deviceId: String = "iphone_16_pro",
        outputDir: URL
    ) throws -> URL {

        let now = ISO8601DateFormatter().string(from: Date())
        let expId = "\(modelInfo.key)_iphone16pro_\(now.prefix(10).replacingOccurrences(of: "-", with: ""))"

        let report: [String: Any] = [
            "schema_version": "1.0.0",
            "experiment_id": expId,
            "device_id": deviceId,
            "model_key": modelInfo.key,
            "model_hf_id": modelInfo.hfId,
            "quantization": modelInfo.quantization,
            "status": "completed",
            "timestamp_start": now,
            "timestamp_end": now,
            "performance_metrics": [
                "ttft_ms_mean": stats.ttftMsMean,
                "ttft_ms_stddev": stats.ttftMsStddev,
                "decode_tokens_per_sec_mean": stats.tpsMean,
                "peak_memory_mb_mean": stats.peakMemMBMean,
                "on_disk_size_mb": modelInfo.onDiskSizeMB,
                "raw_ttft_ms": stats.ttftMsValues,
                "raw_tps": stats.tpsValues,
                "raw_peak_mem_mb": stats.peakMemMBValues
            ],
            "quality_scores": [:],  // quality evaluated separately on Mac
            "notes": "iPhone 16 Pro reference baseline, Phase 0 Task 3.2"
        ]

        let data = try JSONSerialization.data(withJSONObject: report,
                                              options: [.prettyPrinted, .sortedKeys])
        let filename = "\(expId).json"
        let url = outputDir.appendingPathComponent(filename)
        try data.write(to: url)
        return url
    }
}
